import os
import sys
import time
from playwright.sync_api import sync_playwright

def run():
    os.makedirs("screenshots", exist_ok=True)
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        
        # Helper to check horizontal scroll
        def check_no_hscroll(page, label):
            sw = page.evaluate("() => document.documentElement.scrollWidth")
            iw = page.evaluate("() => window.innerWidth")
            print(f"[{label}] Viewport width: {iw}, Document scroll width: {sw}")
            if sw > iw + 2:
                print(f"WARNING: Horizontal scroll detected in {label}: scrollWidth={sw} > innerWidth={iw}")

        def set_theme(page, mode):
            is_dark = page.evaluate("() => document.documentElement.classList.contains('dark')")
            if (mode == "light" and is_dark) or (mode == "dark" and not is_dark):
                btn = page.locator("button[aria-label='Toggle theme']")
                if btn.is_visible():
                    btn.click()
                    page.wait_for_timeout(500)

        # --- 1. Desktop Dark Mode (1280x800) ---
        print("\n--- Testing Screen 1 (Add Data) ---")
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.on("console", lambda msg: console_errors.append(f"CONSOLE {msg.type}: {msg.text} ({msg.location})") if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(f"PAGE ERROR: {err}"))
        page.on("response", lambda res: print(f"HTTP {res.status}: {res.url}") if res.status >= 400 else None)


        page.goto("http://localhost:3000?dataset=sample_credit_card&step=data")
        page.wait_for_timeout(2000)
        
        # Ensure dark theme
        set_theme(page, "dark")
        page.screenshot(path="screenshots/screen1_data_dark.png", full_page=True)
        check_no_hscroll(page, "Screen 1 Dark")

        # Light mode
        print("Capturing Screen 1 Light...")
        set_theme(page, "light")
        page.screenshot(path="screenshots/screen1_data_light.png", full_page=True)
        check_no_hscroll(page, "Screen 1 Light")

        # 768px tablet width
        print("Capturing Screen 1 at 768px...")
        page.set_viewport_size({"width": 768, "height": 900})
        set_theme(page, "dark")
        page.screenshot(path="screenshots/screen1_data_768px.png", full_page=True)
        check_no_hscroll(page, "Screen 1 768px")

        # Restore 1280x900
        page.set_viewport_size({"width": 1280, "height": 900})

        # --- 2. Run Automated Grouping ("Find my customer groups") ---
        print("\n--- Running 'Find my customer groups' ---")
        find_btn = page.locator("button:has-text('Find my customer groups')")
        if find_btn.is_visible():
            find_btn.click()
            print("Clicked 'Find my customer groups'. Waiting for pipeline...")
            page.wait_for_selector("h2:has-text('customer groups')", timeout=60000)
            page.wait_for_timeout(2000)
        else:
            print("Find button not found or already on groups step.")

        # --- 3. Screen 2: Your customer groups ---
        print("\n--- Testing Screen 2 (Your Customer Groups) ---")
        set_theme(page, "dark")
        page.screenshot(path="screenshots/screen2_groups_dark.png", full_page=True)
        check_no_hscroll(page, "Screen 2 Dark")

        print("Capturing Screen 2 Light...")
        set_theme(page, "light")
        page.screenshot(path="screenshots/screen2_groups_light.png", full_page=True)
        check_no_hscroll(page, "Screen 2 Light")

        print("Capturing Screen 2 at 768px...")
        page.set_viewport_size({"width": 768, "height": 900})
        set_theme(page, "dark")
        page.screenshot(path="screenshots/screen2_groups_768px.png", full_page=True)
        check_no_hscroll(page, "Screen 2 768px")

        # --- 4. Screen 3: Take action ---
        print("\n--- Testing Screen 3 (Take Action) ---")
        page.set_viewport_size({"width": 1280, "height": 900})
        # Click "Plan campaign" on first group
        plan_btn = page.locator("button:has-text('Plan campaign')").first
        plan_btn.click()
        page.wait_for_timeout(1500)

        set_theme(page, "dark")
        page.screenshot(path="screenshots/screen3_act_dark.png", full_page=True)
        check_no_hscroll(page, "Screen 3 Dark")

        print("Capturing Screen 3 Light...")
        set_theme(page, "light")
        page.screenshot(path="screenshots/screen3_act_light.png", full_page=True)
        check_no_hscroll(page, "Screen 3 Light")

        print("Capturing Screen 3 at 768px...")
        page.set_viewport_size({"width": 768, "height": 900})
        set_theme(page, "dark")
        page.screenshot(path="screenshots/screen3_act_768px.png", full_page=True)
        check_no_hscroll(page, "Screen 3 768px")

        # --- 5. Methodology View ("How we got these results") ---
        print("\n--- Testing 'How we got these results' ---")
        page.set_viewport_size({"width": 1280, "height": 900})
        set_theme(page, "dark")
        method_btn = page.locator("button:has-text('How we got these results')")
        method_btn.click()
        page.wait_for_timeout(1500)

        page.screenshot(path="screenshots/methodology_simple_dark.png", full_page=True)
        check_no_hscroll(page, "Methodology Simple")

        # Advanced Tab
        adv_tab = page.locator("button:has-text('Advanced details')").first
        adv_tab.click()
        page.wait_for_timeout(1500)
        page.screenshot(path="screenshots/methodology_advanced_dark.png", full_page=True)
        check_no_hscroll(page, "Methodology Advanced Dark")

        print("Capturing Methodology Advanced Light...")
        set_theme(page, "light")
        page.screenshot(path="screenshots/methodology_advanced_light.png", full_page=True)
        check_no_hscroll(page, "Methodology Advanced Light")

        # --- 6. Second CSV Dataset Upload & Testing ---
        print("\n--- Testing Second CSV Dataset (retail_customers.csv) ---")
        page.set_viewport_size({"width": 1280, "height": 900})
        set_theme(page, "dark")
        
        # Go to data step
        page.goto("http://localhost:3000?step=data")
        page.wait_for_timeout(1000)

        # Upload retail_customers.csv via file input
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files("backend/data/retail_customers.csv")
        print("Uploaded retail_customers.csv. Waiting for profile...")
        page.wait_for_timeout(3000)

        page.screenshot(path="screenshots/dataset2_screen1_uploaded.png", full_page=True)
        check_no_hscroll(page, "Dataset 2 Screen 1")

        # Run grouping on second dataset
        find_btn2 = page.locator("button:has-text('Find my customer groups')")
        if find_btn2.is_visible():
            find_btn2.click()
            print("Finding groups for retail_customers.csv...")
            page.wait_for_selector("h2:has-text('customer groups')", timeout=60000)
            page.wait_for_timeout(2000)

            page.screenshot(path="screenshots/dataset2_groups_dark.png", full_page=True)
            check_no_hscroll(page, "Dataset 2 Groups")

            # Go to Act
            plan_btn2 = page.locator("button:has-text('Plan campaign')").first
            plan_btn2.click()
            page.wait_for_timeout(1500)
            page.screenshot(path="screenshots/dataset2_act_dark.png", full_page=True)
            check_no_hscroll(page, "Dataset 2 Act")

        context.close()
        browser.close()

    print("\n--- Console and Page Errors Summary ---")
    if page_errors:
        print("Page Errors encountered:")
        for err in page_errors:
            print("  ", err)
    else:
        print("Zero page errors!")

    if console_errors:
        print("Console errors encountered:")
        for err in console_errors:
            print("  ", err)
    else:
        print("Zero console errors!")

    print("\nAll screenshot captures and responsiveness checks complete!")

if __name__ == "__main__":
    run()
