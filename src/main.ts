import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import minimatch from "minimatch";

import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files
    for (const chunk of file.chunks) {
      const prompt = createPrompt(file, chunk, prDetails);
      console.log("Prompt:", prompt);
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        const newComments = createComment(file, chunk, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

const ReviewCommentSchema = z.object({
  lineNumber: z.string(),
  reviewComment: z.string(),
});

const ReviewSchema = z.object({
  reviews: z.array(ReviewCommentSchema),
});

function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  return `
    You are an expert in Python and Django. Your task is to review pull requests by analyzing the provided code diffs. Please follow these instructions carefully:
    - Respond only in the following JSON format: {"reviews": [{"lineNumber": <line_number>, "reviewComment": "<review comment>"}]}
    - Do not include any positive comments or compliments.
    - Provide comments and suggestions ONLY if there are issues or improvements needed. If there are no issues, "reviews" should be an empty array.
    - Write comments in GitHub Markdown format, focusing solely on code quality, correctness, and best practices.
    - Use the provided pull request title and description only for context related to the code changes.
    - DO NOT suggest adding comments to the code.
    - **Return only valid JSON** without any enclosing backticks or additional formatting.

    Review the following code diff in the file "${
      file.to
    }" considering the pull request title and description:

    Pull request title: ${prDetails.title}
    Pull request description:

    ---
    ${prDetails.description}
    ---

    Git diff to review:

    \`\`\`diff
    ${chunk.content}
    ${chunk.changes
      // @ts-expect-error - ln and ln2 exists where needed
      .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
      .join("\n")}
    \`\`\`
  `;
}

async function getFormattedAIResponse(content: string | null): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  if (!content) return null;

  try {
    const response = await openai.beta.chat.completions.parse({
      model: "gpt-4o-mini",
      response_format: zodResponseFormat(ReviewSchema, "reviews"),
      messages: [
        {
          role: "user",
          content: `Given the following data, format it with the given response format: ${content}`,
        },
      ],
    });
    console.log(
      "AI response (gpt-4o-mini):",
      response.choices[0].message.parsed
    );

    return response.choices[0].message.parsed?.reviews ?? [];
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  try {
    const response = await openai.chat.completions.create({
      model: "o1-preview",
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    });
    console.log(
      "AI response (o1-preview):",
      response.choices[0].message.content
    );

    const formattedResponse = await getFormattedAIResponse(
      response.choices[0].message.content
    );

    return formattedResponse;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  chunk: Chunk,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: Number(aiResponse.lineNumber),
    };
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
  });
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
