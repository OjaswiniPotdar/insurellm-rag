"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  enough_context?: boolean;
  thought_process?: string[];
  streaming?: boolean;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);

    // Add empty assistant message that we'll stream into
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "",
        enough_context: undefined,
        thought_process: [],
        streaming: true,
      },
    ]);

    try {
      const res = await fetch("http://localhost:8000/query/stream",  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No reader");

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.token) {
            // Append token to last message
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = {
                ...last,
                content: last.content + data.token,
              };
              return updated;
            });
          }

          if (data.done) {
            // Mark streaming complete
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = {
                ...last,
                streaming: false,
                enough_context: data.enough_context,
                thought_process: data.thought_process || [],
              };
              return updated;
            });
          }

          if (data.error) {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: "assistant",
                content: "Something went wrong. Please try again.",
                enough_context: false,
                thought_process: [],
                streaming: false,
              };
              return updated;
            });
          }
        }
      }
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Error connecting to the server. Make sure the API is running.",
          enough_context: false,
          thought_process: [],
          streaming: false,
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <main className="min-h-screen bg-[#0f1117] flex flex-col items-center px-4 py-8">

      {/* Header */}
      <div className="w-full max-w-2xl mb-8 text-center">
        <h1 className="text-3xl font-bold text-white tracking-tight">
          🛡️ Insurellm Assistant
        </h1>
        <p className="text-gray-500 text-sm mt-1">
          Ask anything about Insurellm
        </p>
      </div>

      {/* Chat window */}
      <div className="w-full max-w-2xl flex-1 flex flex-col gap-4 mb-6 min-h-[400px]">

        {messages.length === 0 && (
          <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
            Ask a question to get started...
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.role === "user" ? (
              <div className="bg-[#1e2433] border border-[#2d3448] text-gray-200 rounded-2xl rounded-tr-sm px-4 py-3 max-w-[80%] text-sm">
                {msg.content}
              </div>
            ) : (
              <div className="bg-[#162032] border border-[#1e3a5f] border-l-2 border-l-[#c9a84c] rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%] text-sm text-gray-200 flex flex-col gap-2">

                <p>
                  {msg.content}
                  {msg.streaming && (
                    <span className="inline-block w-1.5 h-4 bg-[#c9a84c] ml-0.5 animate-pulse rounded-sm" />
                  )}
                </p>

                {/* Show badges only when streaming is done */}
                {!msg.streaming && (
                  <>
                    <span
                      className={`self-start text-xs px-2 py-0.5 rounded-full font-medium ${
                        msg.enough_context
                          ? "bg-green-900 text-green-300"
                          : "bg-red-900 text-red-300"
                      }`}
                    >
                      {msg.enough_context ? "✓ Context found" : "⚠ Limited context"}
                    </span>

                    {msg.thought_process && msg.thought_process.length > 0 && (
                      <details className="mt-1">
                        <summary className="text-xs text-[#c9a84c] cursor-pointer select-none">
                          View reasoning
                        </summary>
                        <ul className="mt-2 flex flex-col gap-1">
                          {msg.thought_process.map((step, j) => (
                            <li key={j} className="text-xs text-gray-500">
                              • {step}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Initial loading before first token */}
        {loading && messages[messages.length - 1]?.content === "" && (
          <div className="flex justify-start">
            <div className="bg-[#162032] border border-[#1e3a5f] rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-gray-500 flex gap-1 items-center">
              <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: "0ms"}} />
              <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: "150ms"}} />
              <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: "300ms"}} />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="w-full max-w-2xl flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about Insurellm..."
          className="flex-1 bg-[#1e2433] border border-[#2d3448] text-gray-200 rounded-xl px-4 py-3 text-sm outline-none focus:border-[#c9a84c] transition-colors placeholder-gray-600"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="bg-[#c9a84c] hover:bg-[#e8c76a] disabled:opacity-40 disabled:cursor-not-allowed text-[#0f1117] font-semibold rounded-xl px-5 py-3 text-sm transition-colors"
        >
          {loading ? "..." : "Ask"}
        </button>
      </div>

    </main>
  );
}

