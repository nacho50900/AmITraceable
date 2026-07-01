import { Given, Then } from '@cucumber/cucumber'
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
    text && text.includes('Solo se analiza tu propia cuenta de Reddit'),
    `Expected consent box to mention own-account-only analysis, got: "${text}"`,
  )
})

Then('I should see a link to connect with Reddit', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  const link = await page.waitForSelector('a.btn-primary', { timeout: 5000 })
  const href = await link.getAttribute('href')
  assert.ok(
    href && href.includes('/auth/reddit/login'),
    `Expected primary link to point to the Reddit OAuth login route, got: "${href}"`,
  )
})
