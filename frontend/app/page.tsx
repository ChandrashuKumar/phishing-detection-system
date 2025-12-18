"use client";
import { useState } from "react";

interface Result {
  score: number;
  threats: string[];
  domain_age: number;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"email" | "url">("url");
  const [inputValue, setInputValue] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!inputValue.trim()) {
      alert(`Please enter ${activeTab === "email" ? "an email" : "a URL"}`);
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const endpoint =
        activeTab === "email" ? "/api/check-email" : "/api/check-url";
      const body =
        activeTab === "email" ? { email: inputValue } : { url: inputValue };

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();

      if (!res.ok) {
        alert(data.error || "An error occurred");
      } else {
        setResult(data);
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        alert(err.message);
      } else {
        alert("Network error");
      }
    } finally {
      setLoading(false);
    }
  };

  const CircularProgress = ({ score }: { score: number }) => {
    const radius = 70;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 10) * circumference;

    // Get gradient ID based on score
    const getGradientId = (score: number) => {
      if (score >= 7) return "redGradient";
      if (score >= 4) return "orangeGradient";
      return "blueGradient";
    };

    return (
      <div className="relative inline-flex items-center justify-center">
        <svg width="180" height="180" className="transform -rotate-90">
          <defs>
            {/* Light Blue Gradient (Low Risk) */}
            <linearGradient
              id="blueGradient"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#7dd3fc" />
              <stop offset="50%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#0369a1" />
            </linearGradient>
            {/* Orange Gradient (Medium Risk) */}
            <linearGradient
              id="orangeGradient"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#fdba74" />
              <stop offset="50%" stopColor="#fb923c" />
              <stop offset="100%" stopColor="#ea580c" />
            </linearGradient>
            {/* Red Gradient (High Risk) */}
            <linearGradient
              id="redGradient"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#fca5a5" />
              <stop offset="50%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#b91c1c" />
            </linearGradient>
          </defs>
          <circle
            cx="90"
            cy="90"
            r={radius}
            stroke="#f5f5f5"
            strokeWidth="16"
            fill="none"
          />
          <circle
            cx="90"
            cy="90"
            r={radius}
            stroke={`url(#${getGradientId(score)})`}
            strokeWidth="16"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1500 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-bold text-white">
            {score.toFixed(1)}
          </span>
          <span className="text-sm text-gray-300 mt-1">/ 10</span>
        </div>
      </div>
    );
  };

  return (
    <div
      className="min-h-screen bg-white relative overflow-hidden"
      style={{
        backgroundImage: "url(/12356.jpg)",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }}
    >
      {/* Main Content */}
      <div className="relative z-10 container mx-auto px-4 py-12 max-w-4xl">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-300 mb-3">
            Phishing Detection System
          </h1>
          <p className="text-gray-100 text-lg">
            Advanced Machine learning powered phishing detection for emails/text
            messages and URLs
          </p>
        </div>

        {/* Card Container */}
        <div className="bg-white shadow-2xl border-2 border-gray-200 p-8">
          {/* Tabs */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => {
                setActiveTab("url");
                setInputValue("");
                setResult(null);
              }}
              className={`flex-1 py-3 px-6 text-lg font-semibold transition-colors ${
                activeTab === "url"
                  ? "bg-sky-400 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              URL Detection
            </button>
            <button
              onClick={() => {
                setActiveTab("email");
                setInputValue("");
                setResult(null);
              }}
              className={`flex-1 py-3 px-6 text-lg font-semibold transition-colors ${
                activeTab === "email"
                  ? "bg-sky-400 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              Email Detection
            </button>
          </div>

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="mb-8">
            {activeTab === "url" ? (
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Enter URL (e.g., https://example.com)"
                className="w-full px-4 py-4 border-2 border-gray-300 text-lg text-gray-800 placeholder:text-gray-400 focus:outline-none focus:border-sky-400 mb-4"
                disabled={loading}
              />
            ) : (
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Paste email content here..."
                rows={6}
                className="w-full px-4 py-4 border-2 border-gray-300 text-lg text-gray-800 placeholder:text-gray-400 focus:outline-none focus:border-sky-400 mb-4 resize-none"
                disabled={loading}
              />
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-1/3 mx-auto block bg-sky-400 hover:bg-sky-500 text-white font-bold py-4 px-6 text-lg transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {loading
                ? "Analyzing..."
                : `Check ${activeTab === "email" ? "Email" : "URL"}`}
            </button>
          </form>
        </div>

        {/* Results */}
        {result && (
          <div className="mt-8">
            <h2 className="text-3xl font-bold text-white mb-8 text-center">
              Analysis Results
            </h2>

            <div className="flex flex-col items-center mb-8">
              <CircularProgress key={result.score} score={result.score} />
              <div className="mt-4 text-center">
                <p className="text-xl text-gray-200 mb-2">
                  Phishing Risk Score
                </p>
                <div className="flex items-center justify-center gap-3">
                  {/* Pulsing dot based on score */}
                  <div className="relative flex items-center justify-center">
                    <span
                      className={`absolute inline-flex h-4 w-4 rounded-full opacity-75 animate-ping ${
                        result.score >= 7
                          ? "bg-red-400"
                          : result.score >= 4
                          ? "bg-orange-400"
                          : "bg-sky-400"
                      }`}
                    ></span>
                    <span
                      className={`relative inline-flex rounded-full h-3 w-3 ${
                        result.score >= 7
                          ? "bg-red-500"
                          : result.score >= 4
                          ? "bg-orange-500"
                          : "bg-sky-500"
                      }`}
                    ></span>
                  </div>
                  <p className="text-lg text-gray-300">
                    {result.score >= 7
                      ? "High Risk - Likely Phishing"
                      : result.score >= 4
                      ? "Medium Risk - Be Cautious"
                      : "Low Risk - Appears Safe"}
                  </p>
                </div>
              </div>
            </div>

            {/* Threats Section (only for URLs) */}
            {activeTab === "url" &&
              result.threats &&
              result.threats.length > 0 && (
                <div className="bg-red-900 bg-opacity-80 border-2 border-red-500 p-6 mb-4 backdrop-blur-sm">
                  <h3 className="text-xl font-bold text-red-200 mb-3">
                    🚨 Detected Threats
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {result.threats.map((threat: string, idx: number) => (
                      <span
                        key={idx}
                        className="bg-red-500 text-white px-4 py-2 text-base font-semibold"
                      >
                        {threat}
                      </span>
                    ))}
                  </div>
                </div>
              )}

            {/* Domain Age Section (only for URLs) */}
            {activeTab === "url" &&
              result.domain_age !== null &&
              result.domain_age !== undefined && (
                <div className="text-center mt-6">
                  <p className="text-white text-xl font-semibold">
                    Domain Age: {result.domain_age} days (
                    {(result.domain_age / 365).toFixed(1)} years)
                  </p>
                </div>
              )}
          </div>
        )}
      </div>
    </div>
  );
}
