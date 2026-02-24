import { Component, StrictMode } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("React render error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 40, color: "#f87171", background: "#0f0f0f", height: "100vh", fontFamily: "monospace" }}>
          <h1 style={{ color: "#e8e8e8" }}>Something went wrong</h1>
          <pre style={{ marginTop: 16, whiteSpace: "pre-wrap" }}>
            {this.state.error.message}
          </pre>
          <pre style={{ marginTop: 8, fontSize: 12, color: "#888", whiteSpace: "pre-wrap" }}>
            {this.state.error.stack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
);
