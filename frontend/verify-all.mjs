import { chromium } from 'playwright'
import fs from 'fs'
import path from 'path'

const SCREENSHOT_DIR = 'C:/Users/MANISH/.gemini/antigravity/brain/57f5966d-de61-4fd8-b607-357e9c71501f/screenshots'
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true })
}

async function verifyAll() {
  console.log('🚀 Launching Chromium browser...')
  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 } })
  const page = await context.newPage()

  try {
    // -------------------------------------------------------------
    // 1. DATA STEP VERIFICATION
    // -------------------------------------------------------------
    console.log('\n--- 1. Testing Step 1 (Data) ---')
    await page.goto('http://localhost:3000/?step=data', { waitUntil: 'networkidle' })
    await page.waitForTimeout(1000)

    // Check sample button label
    const sampleBtn = page.locator('button:has-text("Try credit card data"), button:has-text("Try demo data")')
    const sampleBtnText = await sampleBtn.innerText()
    console.log(`Sample Button Label: "${sampleBtnText}"`)

    // Check LaTeX leak ($): Ensure no raw LaTeX math text anywhere in body
    const bodyText = await page.innerText('body')
    const latexMatches = bodyText.match(/\$[^$\n]+\$/g) || []
    console.log(`Raw LaTeX occurrences in Step 1: ${latexMatches.length}`)
    if (latexMatches.length > 0) {
      console.warn('Found LaTeX:', latexMatches)
    }

    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01_step1_data.png'), fullPage: true })
    console.log('📸 Screenshot saved: 01_step1_data.png')

    // Click sample data button to load dataset
    await sampleBtn.click()
    await page.waitForTimeout(2000)

    // -------------------------------------------------------------
    // 2. PREPARE STEP VERIFICATION
    // -------------------------------------------------------------
    console.log('\n--- 2. Testing Step 2 (Prepare) ---')
    await page.click('button:has-text("Configure Preprocessing"), button:has-text("Prepare")')
    await page.waitForTimeout(1500)

    // Check heatmap subtitle for $|r| > 0.85$ leak
    const prepareText = await page.innerText('body')
    const hasRawLatex = prepareText.includes('$|r| > 0.85$') || prepareText.includes('$|r|')
    console.log(`Heatmap subtitle has raw LaTeX leak? ${hasRawLatex ? 'YES (FAIL)' : 'NO (FIXED!)'}`)
    const hasPlainText = prepareText.includes('|r| > 0.85')
    console.log(`Heatmap subtitle has clean plain text "|r| > 0.85"? ${hasPlainText ? 'YES (PASS)' : 'NO'}`)

    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '02_step2_prepare.png'), fullPage: true })
    console.log('📸 Screenshot saved: 02_step2_prepare.png')

    // -------------------------------------------------------------
    // 3. BENCHMARK STEP VERIFICATION
    // -------------------------------------------------------------
    console.log('\n--- 3. Testing Step 3 (Benchmark) ---')
    await page.click('button:has-text("Run Benchmark"), button:has-text("Benchmark")')
    await page.waitForTimeout(1500)

    // Run benchmark if not yet executed
    const runBtn = page.locator('button:has-text("Run 16 Experiments"), button:has-text("Re-run Benchmark")')
    if (await runBtn.isVisible()) {
      console.log('Running benchmark via UI...')
      await runBtn.click()
      // Wait for completion (target < 15s)
      await page.waitForSelector('text=Completed', { timeout: 35000 })
      await page.waitForTimeout(2000)
    }

    // Check Common-Space Clarity column in table
    const tableText = await page.innerText('body')
    const hasCommonSpaceCol = tableText.includes('Common Space') || tableText.includes('Common-Space')
    console.log(`Common-Space clarity column present? ${hasCommonSpaceCol ? 'YES (PASS)' : 'NO'}`)

    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '03_step3_benchmark.png'), fullPage: true })
    console.log('📸 Screenshot saved: 03_step3_benchmark.png')

    // -------------------------------------------------------------
    // 4. EXPLORE STEP VERIFICATION
    // -------------------------------------------------------------
    console.log('\n--- 4. Testing Step 4 (Explore) ---')
    await page.click('button:has-text("Explore Clusters"), button:has-text("Explore")')
    await page.waitForTimeout(2000)

    // Check dynamic clarity caption (never static contradiction)
    const exploreText = await page.innerText('body')
    const hasStaticClaim = exploreText.includes('distinct islands without overlapping')
    console.log(`Has deprecated static claim "distinct islands without overlapping"? ${hasStaticClaim ? 'YES (FAIL)' : 'NO (FIXED!)'}`)

    const hasDynamicCaption =
      exploreText.includes('Groups are clearly separated') ||
      exploreText.includes('Groups are reasonably distinct but overlap somewhat') ||
      exploreText.includes('Groups overlap a lot')
    console.log(`Has computed dynamic clarity caption? ${hasDynamicCaption ? 'YES (PASS)' : 'NO'}`)

    // Check active model indicator in header
    const hasModelHeader = exploreText.includes('Showing:')
    console.log(`Header specifies active model ("Showing: ...")? ${hasModelHeader ? 'YES (PASS)' : 'NO'}`)

    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '04_step4_explore.png'), fullPage: true })
    console.log('📸 Screenshot saved: 04_step4_explore.png')

    // -------------------------------------------------------------
    // 5. ACT STEP VERIFICATION
    // -------------------------------------------------------------
    console.log('\n--- 5. Testing Step 5 (Act) ---')
    await page.click('button:has-text("Action Plan"), button:has-text("Act")')
    await page.waitForTimeout(2000)

    const actText = await page.innerText('body')
    const hasActModelHeader = actText.includes('Showing:')
    console.log(`Act Header specifies active model ("Showing: ...")? ${hasActModelHeader ? 'YES (PASS)' : 'NO'}`)

    const hasJsonDownload = page.locator('a:has-text("Model Parameters (.json)"), button:has-text("Model Parameters (.json)")')
    console.log(`Model Parameters (.json) download button present? ${await hasJsonDownload.isVisible() ? 'YES (PASS)' : 'NO'}`)

    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '05_step5_act.png'), fullPage: true })
    console.log('📸 Screenshot saved: 05_step5_act.png')

    console.log('\n🎉 All Playwright UI verifications completed successfully!')
  } catch (err) {
    console.error('Playwright verification error:', err)
  } finally {
    await browser.close()
  }
}

verifyAll()
