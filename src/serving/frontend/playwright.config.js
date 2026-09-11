import { defineConfig } from "@playwright/test";

export default defineConfig({
    testDir: "tests",
    fullyParallel: true,
    workers: process.env.CI ? 2 : undefined,
    forbidOnly: !!process.env.CI,
    retries: 0,
    use: { baseURL: "http://127.0.0.1:4173", trace: "retain-on-failure" },
    webServer: { command: "node tests/server.mjs", url: "http://127.0.0.1:4173", reuseExistingServer: false },
});
