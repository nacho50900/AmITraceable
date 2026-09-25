import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import './i18n'
import Dashboard from './pages/Dashboard'
import Landing from './pages/Landing'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* Sin `future`: era el mecanismo de react-router-dom v6 para adoptar
        pronto comportamientos de v7 (v7_startTransition,
        v7_relativeSplatPath). Con v7 ya instalado (ver package.json),
        esos comportamientos son el default -- la prop `future` ni
        siquiera existe ya en BrowserRouterProps, y dejarla rompía
        `tsc -b` (y por tanto `npm run build`, y por tanto la imagen
        Docker del webapp). */}
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
