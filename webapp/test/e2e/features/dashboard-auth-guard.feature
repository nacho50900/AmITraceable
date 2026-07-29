Feature: Dashboard auth guard
  Validate that the dashboard checks real authentication status against
  the backend (not just a frontend-only check) before showing any report,
  and sends unauthenticated visitors back to the landing page.

  This exercises a real HTTP round-trip to the backend's auth-status
  endpoint (no mocking) -- the kind of integration bug (wrong route, CORS,
  broken session middleware...) that component tests with a mocked `api`
  module cannot catch.

  Scenario: Visiting the dashboard directly without logging in redirects home
    Given I am not logged in to any platform
    When I open the dashboard directly for "reddit"
    Then I should be redirected back to the landing page
