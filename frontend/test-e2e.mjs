import { chromium } from 'playwright'
import fs from 'fs'
import path from 'path'

const SCREENSHOT_DIR = 'C:/Users/MANISH/.gemini/antigravity/brain/7d7da3c2-60ac-430a-ab50-759c0aeb1917/screenshots'
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true })
}

async function runE2E() {
  console.log('🚀 Launching Chromium browser...')
  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
  })

  const page = await context.newPage()

  // Collect console messages and errors
  const consoleErrors = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      console.log('❌ Browser Console Error:', msg.text())
      consoleErrors.push(msg.text())
    } else {
      console.log('ℹ️ Console:', msg.text())
    }
  })

  page.on('pageerror', (err) => {
    console.log('💥 Page Error:', err.message)
    consoleErrors.push(err.message)
  })

  console.log('🌐 Navigating to http://localhost:3000...')
  await page.goto('http://localhost:3000?step=data', { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForTimeout(1500)

  async function toggleTheme() {
    const btn = await page.$('button[aria-label="Toggle theme"]')
    if (btn) {
      await btn.click()
      await page.waitForTimeout(500)
    }
  }

  // ---------------- STEP 1: DATA (DARK MODE) ----------------
  console.log('📸 Capturing Step 1: Data (Dark Mode)...')
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step1_data_dark.png'), fullPage: true })

  // ---------------- STEP 1: DATA (LIGHT MODE) ----------------
  console.log('☀️ Toggling Light Mode...')
  await toggleTheme()
  console.log('📸 Capturing Step 1: Data (Light Mode)...')
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step1_data_light.png'), fullPage: true })
  // Toggle back to dark mode
  await toggleTheme()

  // ---------------- HELP DRAWER WITH METHODOLOGY ----------------
  console.log('📖 Opening Help Drawer...')
  const helpBtn = await page.$('button[title*="Help" i]')
  if (helpBtn) {
    await helpBtn.click()
    await page.waitForTimeout(600)
    console.log('📸 Capturing Help Drawer (Methodology)...')
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'help_drawer_methodology.png'), fullPage: true })
    const closeHelp = await page.$('button:has-text("Got it, close")') || await page.$('button:has(svg.lucide-x)')
    if (closeHelp) await closeHelp.click()
    await page.waitForTimeout(400)
  }

  // ---------------- STEP 2: PREPARE ----------------
  console.log('➡️ Advancing to Step 2: Prepare...')
  await page.goto('http://localhost:3000?step=prepare', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(1500)
  console.log('📸 Capturing Step 2: Prepare (Dark Mode)...')
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step2_prepare_dark.png'), fullPage: true })

  // Light mode for Step 2
  await toggleTheme()
  console.log('📸 Capturing Step 2: Prepare (Light Mode)...')
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step2_prepare_light.png'), fullPage: true })
  await toggleTheme()

  // ---------------- STEP 3: BENCHMARK ----------------
  console.log('➡️ Advancing to Step 3: Benchmark...')
  await page.goto('http://localhost:3000?step=benchmark', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(1500)
  console.log('📸 Capturing Step 3: Benchmark (Dark Mode)...')
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step3_benchmark_dark.png'), fullPage: true })

  // Toggle Expert View in Benchmark
  console.log('🔍 Toggling Expert View in Step 3...')
  const expertBtn = (await page.$$('button:has-text("Expert View")'))[0]
  if (expertBtn) {
    await expertBtn.click()
    await page.waitForTimeout(600)
    console.log('📸 Capturing Step 3: Benchmark Expert View...')
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step3_benchmark_expert.png'), fullPage: true })
  }

  // ---------------- STEP 4: EXPLORE ----------------
  console.log('➡️ Advancing to Step 4: Explore...')
  await page.goto('http://localhost:3000?step=explore', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(1500)
  console.log('📸 Capturing Step 4: Explore (Dark Mode)...')
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step4_explore_dark.png'), fullPage: true })

  // ---------------- STEP 5: ACT ----------------
  console.log('➡️ Advancing to Step 5: Act...')
  await page.goto('http://localhost:3000?step=act', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(1500)
  console.log('📸 Capturing Step 5: Act (Dark Mode)...')
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step5_act_dark.png'), fullPage: true })

  // Open Campaign Drawer
  console.log('📋 Opening Campaign Drawer in Step 5...')
  const planCampaignBtn = (await page.$$('button:has-text("Plan campaign")'))[0]
  if (planCampaignBtn) {
    await planCampaignBtn.click()
    await page.waitForTimeout(800)
    console.log('📸 Capturing Step 5: Campaign Drawer & ROI Simulator...')
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step5_campaign_drawer.png'), fullPage: true })

    // Close drawer
    const closeBtn = await page.$('button:has-text("Done")')
    if (closeBtn) await closeBtn.click()
    await page.waitForTimeout(400)
  }

  // Open Customer Scoring Modal
  console.log('🎯 Opening Customer Scoring Modal...')
  const scoreBtn = await page.$('button:has-text("Score One Customer")')
  if (scoreBtn) {
    await scoreBtn.click()
    await page.waitForTimeout(600)
    console.log('📸 Capturing Step 5: Customer Scoring Modal...')
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'step5_score_modal.png'), fullPage: true })
    const closeBtn = await page.$('button:has-text("×")') || (await page.$$('form button'))[0]
  }

  // ---------------- SWITCH TO SECOND DATASET (RFM) ----------------
  console.log('\n🔄 Switching to Second Dataset: ecommerce_rfm_test.csv...')
  await page.goto('http://localhost:3000?step=data', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(500)

  // Click dataset dropdown in top bar
  const datasetDropdownBtn = (await page.$$('header button'))[0]
  if (datasetDropdownBtn) {
    await datasetDropdownBtn.click()
    await page.waitForTimeout(600)
    const rfmOption = (await page.$$('button:has-text("ecommerce_rfm_test")'))[0]
    if (rfmOption) {
      await rfmOption.click()
      await page.waitForTimeout(1500)
      console.log('📸 Capturing RFM Dataset Step 1: Data...')
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'rfm_step1_data.png'), fullPage: true })

      // Advance through Prepare & Benchmark for RFM
      const prepareBtn = (await page.$$('button:has-text("Looks good, prepare data")'))[0]
      if (prepareBtn) {
        await prepareBtn.click()
        await page.waitForTimeout(1500)
        const runBenchmarkBtn = (await page.$$('button:has-text("Run 16-experiment benchmark")'))[0]
        if (runBenchmarkBtn) {
          await runBenchmarkBtn.click()
          await page.waitForTimeout(4000)
        }
      }

      const exploreBtn = (await page.$$('button:has-text("Explore customer segments")'))[0]
      if (exploreBtn) {
        await exploreBtn.click()
        await page.waitForTimeout(1000)
        const actBtn = (await page.$$('button:has-text("Plan campaigns & actions")'))[0]
        if (actBtn) {
          await actBtn.click()
          await page.waitForTimeout(1500)
        }
      }

      console.log('📸 Capturing RFM Dataset Step 5: Act (E-Commerce Personas)...')
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'rfm_step5_act.png'), fullPage: true })
    }
  }

  console.log('\n🏁 E2E Test Run Completed!')
  console.log(`Total Console Errors: ${consoleErrors.length}`)
  if (consoleErrors.length > 0) {
    console.log('Errors:', consoleErrors)
  }

  await browser.close()
}

runE2E().catch((err) => {
  console.error('E2E Runner Failed:', err)
  process.exit(1)
})
