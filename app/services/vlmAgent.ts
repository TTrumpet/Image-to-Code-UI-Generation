import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Readable } from "node:stream";

const PYTHON_TIMEOUT_MS = 3_000_000; // 5 minutes
const PYTHON_SCRIPT = "run_agent_once.py";

type AgentAttachment = {
  id: string;
  name: string;
  type: "image" | "file";
  previewUrl?: string;
  size?: number;
};

type PythonAgentSuccess = {
  success: true;
  result: {
    plan?: string;
    reasoning_steps?: string | string[];
    code?: string;
    feedback?: string;
    error?: string | null;
    logs?: string;
    preview_image?: {
      name?: string;
      dataUrl: string;
      size?: number;
    };
    final_html_path?: string;
    final_html_web_path?: string | null;
  };
  logs?: string;
};

type PythonAgentError = {
  success: false;
  error: string;
  logs?: string;
};

type PythonAgentResponse = PythonAgentSuccess | PythonAgentError;

export type AgentChatMessage = {
  id: string;
  role: "assistant";
  variant: "accent" | "subtle";
  content: string;
  attachments?: AgentAttachment[];
  htmlPath?: string;
  htmlWebPath?: string | null;
};

export type AgentRunResult = {
  messages: AgentChatMessage[];
  status: {
    kind: "success" | "error";
    text: string;
    detail?: string;
  };
  usedFallback: boolean;
  finalHtmlPath?: string;
  finalHtmlWebPath?: string | null;
};

const serviceDir = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = resolve(serviceDir, "..", "..");
export const pythonWorkingDir = resolve(repoRoot, "Image-to-Code-UI-Generation");
const pythonScriptPath = resolve(pythonWorkingDir, PYTHON_SCRIPT);

function collectStream(stream: Readable): Promise<string> {
  return new Promise((resolveStream) => {
    const chunks: Buffer[] = [];
    stream.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    stream.on("end", () => resolveStream(Buffer.concat(chunks).toString("utf8")));
  });
}

async function runPythonProcess(prompt: string, imagePath?: string): Promise<PythonAgentResponse> {
  const pythonExecutable = process.env.PYTHON_BINARY ?? process.env.PYTHON ?? "python";

  return await new Promise<PythonAgentResponse>((resolveResult) => {
    let resolved = false;

    let subprocess: ChildProcessWithoutNullStreams;
    try {
      const env = {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
      };
      subprocess = spawn(pythonExecutable, [pythonScriptPath], {
        cwd: pythonWorkingDir,
        stdio: "pipe",
        env,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown spawn error";
      resolveResult({ success: false, error: `Failed to start python process: ${message}` });
      return;
    }

    const timeout = setTimeout(() => {
      if (resolved) {
        return;
      }
      resolved = true;
      subprocess.kill();
      resolveResult({ success: false, error: "Python agent timed out." });
    }, PYTHON_TIMEOUT_MS);

    const stdoutPromise = collectStream(subprocess.stdout);
    const stderrPromise = collectStream(subprocess.stderr);

    void Promise.all([stdoutPromise, stderrPromise]).then(([stdout, stderr]) => {
      if (resolved) {
        return;
      }

      resolved = true;
      clearTimeout(timeout);

      try {
        const parsed = JSON.parse(stdout) as PythonAgentResponse;
        if (stderr) {
          if (parsed.success) {
            parsed.result.logs = parsed.result.logs
              ? `${parsed.result.logs}\n${stderr}`
              : stderr;
          } else {
            parsed.logs = parsed.logs ? `${parsed.logs}\n${stderr}` : stderr;
          }
        }
        resolveResult(parsed);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown JSON parse error";
        const combinedLogs = [stdout, stderr].filter(Boolean).join("\n\n");
        resolveResult({
          success: false,
          error: `Unable to parse python output: ${message}`,
          logs: combinedLogs || undefined,
        });
      }
    });

    subprocess.on("error", (error) => {
      if (resolved) {
        return;
      }
      resolved = true;
      clearTimeout(timeout);
      const message = error instanceof Error ? error.message : "Unknown process error";
      resolveResult({ success: false, error: `Python process error: ${message}` });
    });

    subprocess.stdin.write(
      JSON.stringify({
        prompt,
        image_path: imagePath ?? null,
      }),
      "utf8",
      () => {
        subprocess.stdin.end();
      },
    );
  });
}

function buildMessagesFromPython(result: PythonAgentSuccess["result"]): AgentChatMessage[] {
  const messages: AgentChatMessage[] = [];
  if (result.plan) {
    messages.push({
      id: randomUUID(),
      role: "assistant",
      variant: "accent",
      content: `Here's the plan I'll follow:\n${result.plan}`.trim(),
    });
  }

  const reasoningSteps = Array.isArray(result.reasoning_steps)
    ? result.reasoning_steps
    : typeof result.reasoning_steps === "string" && result.reasoning_steps.length > 0
      ? result.reasoning_steps.split("\n")
      : undefined;

  if (reasoningSteps && reasoningSteps.length > 0) {
    messages.push({
      id: randomUUID(),
      role: "assistant",
      variant: "subtle",
      content: `Detailed steps:\n${reasoningSteps.join("\n")}`,
    });
  }

  if (result.code) {
    messages.push({
      id: randomUUID(),
      role: "assistant",
      variant: "accent",
      content: `Generated Python snippet:\n${result.code.trim()}`,
      htmlPath: result.final_html_path,
      htmlWebPath: result.final_html_web_path,
    });
  }

  if (result.preview_image?.dataUrl) {
    messages.push({
      id: randomUUID(),
      role: "assistant",
      variant: "accent",
      content: "Here's the rendered preview.",
      attachments: [
        {
          id: randomUUID(),
          name: result.preview_image.name ?? "agent-preview.png",
          type: "image",
          previewUrl: result.preview_image.dataUrl,
          size: result.preview_image.size,
        },
      ],
    });
  }

  const feedback = result.feedback ?? "";
  if (feedback) {
    messages.push({
      id: randomUUID(),
      role: "assistant",
      variant: "subtle",
      content: `Latest feedback: ${feedback}`,
    });
  }

  const errorNote = result.error ?? "";
  if (errorNote) {
    messages.push({
      id: randomUUID(),
      role: "assistant",
      variant: "subtle",
      content: `Executor reported an issue: ${errorNote}`,
    });
  }

  if (messages.length === 0) {
    messages.push({
      id: randomUUID(),
      role: "assistant",
      variant: "subtle",
      content: "The agent completed without generating any updates.",
    });
  }

  return messages;
}

function fallbackMessages(prompt: string): AgentRunResult {
  const templateCode =
    [
      "from PIL import Image, ImageDraw",
      "",
      "# Mocked fallback pipeline—replace with model outputs in production.",
      "img = Image.new('RGB', (800, 600), 'white')",
      "draw = ImageDraw.Draw(img)",
      "draw.text((40, 40), 'Image2Code fallback render', fill='black')",
      "img.save('fallback-output.png')",
    ].join("\n");

  const messages: AgentChatMessage[] = [
    {
      id: randomUUID(),
      role: "assistant",
      variant: "accent",
      content: `I received your request:\n"${prompt}"\n\nHere's a high-level plan:\n1. Break down the UI into key regions.\n2. Translate those into layout primitives.\n3. Map typography and colors from the input.`,
    },
    {
      id: randomUUID(),
      role: "assistant",
      variant: "subtle",
      content:
        "Detailed steps:\n- Analyze the primary call-to-action and hero layout.\n- Select component primitives (cards, list rows, nav bars).\n- Produce semantic HTML or React code with Tailwind annotations.",
    },
    {
      id: randomUUID(),
      role: "assistant",
      variant: "accent",
      content: `Generated Python snippet:\n${templateCode}`,
    },
    {
      id: randomUUID(),
      role: "assistant",
      variant: "subtle",
      content:
        "Latest feedback: This is a mocked response path because the Python agent was unavailable. Swap in real model integration when the service is online.",
    },
  ];

  return {
    messages,
    status: {
      kind: "error",
      text: "Fallback pathway used because the Python VLM agent is unavailable.",
    },
    usedFallback: true,
  };
}

export async function runVlmAgent(prompt: string, imagePath?: string): Promise<AgentRunResult> {
  const pythonResult = await runPythonProcess(prompt, imagePath);

  if (!pythonResult.success) {
    const errorMessage = pythonResult.error ?? "The VLM agent failed without providing a reason.";

    console.error("________________________________________");
    console.error("PYTHON AGENT FAILED");
    console.error("Error Message:", errorMessage);
    if ('logs' in pythonResult && pythonResult.logs) {
        console.error("Python Logs/Traceback:", pythonResult.logs);
    }
    console.error("________________________________________");

    if (shouldUseFallback(errorMessage)) {
      return fallbackMessages(prompt);
    }
    return {
      messages: [],
      status: {
        kind: "error",
        text: "The VLM agent could not complete the request.",
        detail: errorMessage,
      },
      usedFallback: false,
    };
  }

  const messages = buildMessagesFromPython(pythonResult.result);

  return {
    messages,
    status: {
      kind: "success",
      text: "VLM agent generated a fresh response.",
      detail: pythonResult.logs ?? pythonResult.result.logs,
    },
    usedFallback: false,
    finalHtmlPath: pythonResult.result.final_html_path,
    finalHtmlWebPath: pythonResult.result.final_html_web_path,
  };
}

function shouldUseFallback(errorMessage: string): boolean {
  const lower = errorMessage.toLowerCase();
  if (lower.includes("failed to start python process")) {
    return true;
  }
  if (lower.includes("python agent timed out") || lower.includes("timed out")) {
    return true;
  }
  if (lower.includes("unable to parse python output")) {
    return true;
  }
  if (lower.includes("python process error")) {
    return true;
  }
  return false;
}
