import { Given, Then, When } from '@cucumber/cucumber'
import assert from 'assert'

Given('I am not logged in to any platform', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  // Cada escenario arranca con un contexto de navegador nuevo (ver
  // support/setup.mjs), así que normalmente ya no hay ninguna cookie de
  // sesión -- se limpia explícitamente igualmente, para que este step no
  // dependa en silencio de ese detalle de implementación del hook.
  await page.context().clearCookies()
})

When('I open the dashboard directly for {string}', async function (platform) {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  await page.goto(`http://localhost:5173/dashboard?platform=${platform}`)
})

Then('I should be redirected back to the landing page', async function () {
  const page = this.page
  if (!page) throw new Error('Page not initialized')
  // El redirect depende de una llamada real (sin mockear) al backend
  // (api.authStatus), así que hay que esperar a la navegación, no
  // comprobar la URL al instante.
  await page.waitForURL('http://localhost:5173/', { timeout: 10_000 })
  assert.strictEqual(new URL(page.url()).pathname, '/')
})
