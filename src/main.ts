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

// Fetch PR metadata
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

// Fetch the PR diff (or compare commits) as a raw string
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

// Gather all diff chunks into an array so we can pass them in a single request
interface DiffChunkData {
  chunkIndex: number;
  filePath: string;
  diffText: string; // The combined chunk content
}

function gatherDiffData(parsedDiff: File[]): DiffChunkData[] {
  const diffChunkData: DiffChunkData[] = [];
  let globalIndex = 0;

  for (const file of parsedDiff) {
    // ignore deleted files
    if (file.to === "/dev/null") continue;

    /**
     * Create a diff string for each chunk resembling the following:
     * @@ -17,5 +17,4 @@ jobs:
        17    uses: pace-systems/ai-code-reviewer
        18    with:
        19        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
     */
    for (const chunk of file.chunks) {
      const chunkDiffText = [
        chunk.content,
        ...chunk.changes.map(
          // @ts-expect-error - ln and ln2 exists where needed
          (c) => `${c.ln ? c.ln : c.ln2} ${c.content}`
        ),
      ].join("\n");

      diffChunkData.push({
        chunkIndex: globalIndex++,
        filePath: file.to || "",
        diffText: chunkDiffText,
      });
    }
  }
  return diffChunkData;
}

// Build one large prompt with all chunk data
function createPrompt(
  prDetails: PRDetails,
  diffChunks: DiffChunkData[]
): string {
  /**
   * Ask the model to return an array of objects. Each object has:
   * {
   *   chunkIndex: <the chunkIndex from diffChunks[]>,
   *   reviews: [ { lineNumber: string, reviewComment: string }, ... ]
   * }
   * If no issues for a chunk, reviews is an empty array.
   * For clarity, label each chunk with its index and the path, plus the PR info.
   */
  let allDiffsSection = "";
  for (const chunk of diffChunks) {
    allDiffsSection += `
      -- Chunk #${chunk.chunkIndex} (File: ${chunk.filePath}) --
      \`\`\`diff
      ${chunk.diffText}
      \`\`\`
    `;
  }

  return `
    You are an expert in Python and Django. Your task is to review the entire pull request by analyzing the code diffs below. Please follow these instructions carefully:

    - Return your response ONLY in valid JSON, with no backticks or extra text, as an array of objects:
    [
      {
        "chunkIndex": number,
        "reviews": [
          { "lineNumber": string, "reviewComment": string },
          ...
        ]
      },
      ...
    ]

    - DO NOT suggest adding code comments.
    - Do NOT include any positive comments or compliments.
    - Provide comments and suggestions ONLY if there are issues or improvements needed.
    - If a chunk has no issues, make "reviews" an empty array for that chunk.
    - Write each "reviewComment" in GitHub Markdown format, focusing on code quality, correctness, and best practices.

    Pull Request Title: ${prDetails.title}
    Pull Request Description:
    ${prDetails.description}

    Now review each chunk below. The chunks are labeled with "chunkIndex" so you can reference them in your JSON output:

    ${allDiffsSection}
  `;
}

async function getAIResponse(prompt: string): Promise<string | null> {
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
    // Return the plain text from the model
    const content = response.choices[0].message?.content || "";
    return content.trim();
  } catch (error) {
    console.error("Error (o1-preview):", error);
    return null;
  }
}

const ReviewCommentSchema = z.object({
  lineNumber: z.string(),
  reviewComment: z.string(),
});

const FullReviewSchema = z.array(
  z.object({
    chunkIndex: z.number(),
    reviews: z.array(ReviewCommentSchema),
  })
);

async function getFormattedAIResponse(rawContent: string | null): Promise<
  Array<{
    chunkIndex: number;
    reviews: Array<{ lineNumber: string; reviewComment: string }>;
  }>
> {
  if (!rawContent) return [];

  try {
    const response = await openai.beta.chat.completions.parse({
      model: "gpt-4o-mini",
      response_format: zodResponseFormat(FullReviewSchema, "reviews"),
      messages: [
        {
          role: "user",
          content: `Format the following text as valid JSON that matches the given schema: ${rawContent}`,
        },
      ],
    });

    const parsed = response.choices[0].message.parsed || [];
    return parsed;
  } catch (error) {
    console.error("Error (gpt-4o-mini):", error);
    return [];
  }
}

// Convert the final chunk-based JSON to GitHub comment objects
function mapReviewsToComments(
  diffChunks: DiffChunkData[],
  allReviews: Array<{
    chunkIndex: number;
    reviews: Array<{ lineNumber: string; reviewComment: string }>;
  }>
): Array<{ body: string; path: string; line: number }> {
  // Build a map chunkIndex -> file path
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const item of allReviews) {
    const chunkData = diffChunks.find((c) => c.chunkIndex === item.chunkIndex);
    if (!chunkData) continue;

    // For each line comment, create the GH comment
    for (const review of item.reviews) {
      comments.push({
        body: review.reviewComment,
        path: chunkData.filePath,
        line: Number(review.lineNumber),
      });
    }
  }

  return comments;
}

// Post the combined review
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

// Main entry point
async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  // Handle PR "opened" or "synchronize"
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

  // Parse the raw diff into structured chunks
  const parsedDiff = parseDiff(diff);

  // Respect "exclude" patterns from the action input
  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  // Gather all chunk data
  const diffChunkData = gatherDiffData(filteredDiff);
  if (diffChunkData.length === 0) {
    console.log("No chunks to analyze after excluding patterns.");
    return;
  }

  // Create one big prompt for the entire diff set
  const prompt = createPrompt(prDetails, diffChunkData);
  console.log("Prompt:", prompt);

  // Single request to o1-preview
  const rawAiResponse = await getAIResponse(prompt);
  console.log("Raw AI response (o1-preview):", rawAiResponse);

  // Single chained request to gpt-4o-mini to ensure valid JSON structure
  const allReviews = await getFormattedAIResponse(rawAiResponse);
  console.log("Formatted AI response (gpt-4o-mini):", allReviews);

  // Map final reviews to GH comment objects
  const comments = mapReviewsToComments(diffChunkData, allReviews);

  // Post them as a single review if there are any
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

// Run the action
main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
