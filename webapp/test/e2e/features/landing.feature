Feature: Landing page consent screen
  Validate that the identity exposure tool shows an explicit consent
  screen before letting the user connect their Instagram account.
  (Reddit is temporarily disabled -- see PLATFORM_CARDS in Landing.tsx --
  so it is not covered by this scenario.)

  Scenario: Consent screen is shown before authentication
    Given the landing page is open
    Then I should see the consent notice
    When I select the "Instagram" card
    Then I should see a link to connect with Instagram

  Scenario: The X card is marked as "Coming Soon" and its CTA is disabled
    Given the landing page is open
    When I select the "X" card
    Then I should see the X card marked as "Coming Soon"
    And the connect button should be disabled and say "Próximamente"
