Feature: Landing page consent screen
  Validate that the identity exposure tool shows an explicit consent
  screen before letting the user connect their Reddit account.

  Scenario: Consent screen is shown before authentication
    Given the landing page is open
    Then I should see the consent notice
    And I should see a link to connect with Reddit
