# E2E tests (Playwright + Cucumber)

This project includes a simple BDD-style E2E setup that uses Playwright for
browser automation and Cucumber (`@cucumber/cucumber`) for Gherkin feature
files.

Quick commands:

- Install Playwright browsers (once):

  ```bash
  npm run test:e2e:install-browsers
  ```

- Run E2E tests (requires both the webapp and the `users` backend running).
  The backend is a Python/FastAPI service, so it's started separately from
  the Node tooling:

  - Start both servers and run the tests automatically:

    ```bash
    npm run test:e2e
    ```

    This uses `start-server-and-test` + `concurrently` to start `npm run dev`
    and `uvicorn app.main:app --port 3000` (see `start:all` in
    `package.json`), wait for `http://localhost:5173`, then run Cucumber.

  - Or start each server yourself in separate terminals and then run:

    ```bash
    npm run test:e2e:run
    ```

Files of interest:
- `test/e2e/features/landing.feature` — consent-screen scenario
- `test/e2e/steps` — step definitions
- `test/e2e/support` — Cucumber World and Playwright hooks

Notes:
- For CI, ensure Playwright browsers are installed (e.g. `npx playwright install --with-deps`).
- The `users` backend needs its Python dependencies installed
  (`pip install -r requirements.txt` inside `users/`, plus the spaCy models —
  see `users/README` section in the main `README.md`) before `start:all` can
  launch it successfully.
- Because the login flow depends on real Reddit OAuth credentials, this E2E
  scenario only checks that the consent screen and the "connect" link are
  rendered — it does not attempt the full OAuth round-trip.
