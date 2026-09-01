import { Given, Then, When } from '@cucumber/cucumber'
import assert from 'assert'

Given('the landing page is open', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  await page.goto('http://localhost:5173')
})

// Orden real de las tarjetas en el carrusel -- ver PLATFORM_CARDS en
// src/pages/Landing.tsx. Se usa para mapear un nombre a su .carousel-dot
// en vez de depender del texto traducido del aria-label (que cambiaría
// con el idioma) o de hacer clic directamente en la tarjeta: las tarjetas
// no activas quedan mayormente tapadas por la activa (transform/zIndex en
// cardStyle()), así que un clic ahí es frágil -- el dot es la forma
// fiable de seleccionar plataforma, tanto para el test como para un
// usuario real.
const CARD_ORDER = ['Reddit', 'Instagram', 'X']

Then('I should see the consent notice', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  await page.waitForSelector('.consent-box', { timeout: 5000 })
  const text = await page.textContent('.consent-box')
  assert.ok(
    text && text.includes('Solo se analiza tu propia cuenta'),
    `Expected consent box to mention own-account-only analysis, got: "${text}"`,
  )
})

Then('I should see a link to connect with Instagram', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  const link = await page.waitForSelector('a.deck-cta', { timeout: 5000 })
  const href = await link.getAttribute('href')
  assert.ok(
    href && href.includes('/auth/instagram/login'),
    `Expected the active card's CTA to link to the Instagram OAuth login route, got: "${href}"`,
  )
})

When('I select the {string} card', async function (cardName) {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  const index = CARD_ORDER.indexOf(cardName)
  if (index === -1) throw new Error(`Unknown platform card: "${cardName}"`)
  const dots = await page.$$('.carousel-dot')
  await dots[index].click()
})

Then('I should see the X card marked as {string}', async function (badgeText) {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  const badge = await page.waitForSelector('.platform-card--x .coming-soon-badge', { timeout: 5000 })
  const text = await badge.textContent()
  assert.strictEqual(text?.trim(), badgeText)
})

Then('the connect button should be disabled and say {string}', async function (buttonText) {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  const button = await page.waitForSelector('.deck-cta', { timeout: 5000 })
  const [tagName, isDisabled, text] = await Promise.all([
    button.evaluate((el) => el.tagName.toLowerCase()),
    button.evaluate((el) => el.disabled === true),
    button.textContent(),
  ])
  assert.strictEqual(tagName, 'button', 'Expected the disabled CTA to be a <button>, not a clickable <a>')
  assert.ok(isDisabled, 'Expected the CTA button to have the disabled attribute')
  assert.ok(text && text.includes(buttonText), `Expected CTA text to include "${buttonText}", got: "${text}"`)
})
