import { Given, Then, When } from '@cucumber/cucumber'
import assert from 'assert'

Given('the landing page is open', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  await page.goto('http://localhost:5173')
})

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

Then('I should see a link to connect with Reddit', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  const link = await page.waitForSelector('a.platform-card--reddit', { timeout: 5000 })
  const href = await link.getAttribute('href')
  assert.ok(
    href && href.includes('/auth/reddit/login'),
    `Expected the Reddit card to link to the Reddit OAuth login route, got: "${href}"`,
  )
})

When('I select the X card', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  const card = await page.waitForSelector('.platform-card--x', { timeout: 5000 })
  await card.click()
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
