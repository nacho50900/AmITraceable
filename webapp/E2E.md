# E2E tests (Playwright + Cucumber)

This project includes a simple BDD-style E2E setup that uses Playwright for
browser automation and Cucumber (`@cucumber/cucumber`) for Gherkin feature
files.

Quick commands:

- Install Playwright browsers (once):

  ```bash
  npm run test:e2e:install-browsers
  ```

- Run E2E tests (requires both the webapp and the `backend` service running).
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
- `test/e2e/features/landing.feature` — consent-screen scenario, and the X card's "Coming Soon" state
- `test/e2e/features/dashboard-auth-guard.feature` — visiting `/dashboard` without logging in redirects home (hits the real backend auth-status endpoint, not mocked)
- `test/e2e/steps` — step definitions
- `test/e2e/support` — Cucumber World and Playwright hooks

Notes:
- For CI, ensure Playwright browsers are installed (e.g. `npx playwright install --with-deps`).
- The `backend` needs its Python dependencies installed
  (`pip install -r requirements.txt` inside `backend/`, plus the spaCy models —
  see `backend/README` section in the main `README.md`) before `start:all` can
  launch it successfully.
- Because the login flow depends on real Reddit OAuth credentials, these E2E
  scenarios only check the consent screen, the "connect" link, the disabled
  "Coming Soon" state, and the auth guard on `/dashboard` — they don't attempt
  the full OAuth round-trip.
- This setup runs the webapp and backend as loose dev servers
  (`npm run dev` + bare `uvicorn`), not the Docker/nginx-proxy setup used in
  `docker-compose.yml` — it won't catch bugs that only show up in the built
  Docker images (e.g. a missing dependency in the backend image).
