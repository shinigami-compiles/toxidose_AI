/* =========================================
   ToxiDose AI - Core Interaction Logic
   ========================================= */

document.addEventListener('DOMContentLoaded', function () {
    // Always start with loading overlay hidden (handles hard reloads)
    hideLoading();

    // Handle browser back/forward cache: when page is restored, hide loader again
    window.addEventListener('pageshow', function () {
        hideLoading();
    });

    // 1. Background animation (pharma icons)
    initBackgroundAnimation();

    // 2. Scroll reveal animations
    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll();

    // 3. CTA navigation + loading (Start Analysis buttons)
    setupCTANavigationWithLoading();

    // 4. Fix "Home" links that point to /base -> redirect to /
    normalizeHomeLinks();

    // 5. Enhance CSV file input label
    enhanceFileInput();

    // 6. Default tab on Assessment Page (Manual tab)
    const manualTabSection = document.getElementById('manual');
    if (manualTabSection) {
        openTab(null, 'manual');
    }

    // 7. Setup Analyze Risk form loading (delay submit by 2s with overlay)
    setupFormLoading();

    // 8. FAQ toggles (help page)
    setupFAQToggle();
});


/* =========================================
   GLOBAL LOADING OVERLAY HELPERS
   ========================================= */

let activeLoadingButton = null;

function showLoading(message, button) {
    const overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        // If overlay HTML not present, just return (no crash)
        return;
    }

    const msgEl = overlay.querySelector('.loading-message');
    if (msgEl && message) {
        msgEl.textContent = message;
    }

    if (button) {
        activeLoadingButton = button;
        button.classList.add('is-loading');
    }

    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (!overlay) return;

    overlay.classList.remove('active');

    if (activeLoadingButton) {
        activeLoadingButton.classList.remove('is-loading');
        activeLoadingButton = null;
    }
}

/* =========================================
   1. BACKGROUND ANIMATION (Roaming Pharma Icons)
   ========================================= */

function initBackgroundAnimation() {
    const container = document.getElementById('background-animation');
    if (!container) return;

    // Medical & Pharmaceutical Icons
    const icons = ['💊', '💉', 'Rx', '🧪', '⚕️', '🩺', '🩸', '🦠', '🧬', '🔬', '🏥', '💓'];
    const count = 35;

    for (let i = 0; i < count; i++) {
        const el = document.createElement('div');
        el.classList.add('floating-item');
        el.innerText = icons[Math.floor(Math.random() * icons.length)];

        // Random Position
        el.style.left = Math.random() * 100 + 'vw';
        el.style.top = 100 + Math.random() * 80 + 'vh'; // start below viewport

        // Duration and Delay
        const duration = 18 + Math.random() * 18; // 18–36s
        const delay = -Math.random() * duration;
        el.style.animationDuration = duration + 's';
        el.style.animationDelay = delay + 's';

        // Varied Size
        const size = 1.6 + Math.random() * 1.4; // rem
        el.style.fontSize = size + 'rem';

        container.appendChild(el);
    }
}

/* =========================================
   2. SCROLL REVEAL
   ========================================= */

function revealOnScroll() {
    const reveals = document.querySelectorAll(
        '.feature-card, .process-step, .info-page-container, ' +
        '.form-container, .results-container, .batch-results-container, ' +
        '.help-section, .summary-card, .risk-card'
    );
    const windowHeight = window.innerHeight;
    const elementVisible = 120;

    for (let i = 0; i < reveals.length; i++) {
        const elementTop = reveals[i].getBoundingClientRect().top;
        if (elementTop < windowHeight - elementVisible) {
            reveals[i].classList.add('animate-up');
        }
    }
}

/* =========================================
   3. CTA & NAVIGATION BEHAVIOR (with loading)
   ========================================= */

function setupCTANavigationWithLoading() {
    const ctaButtons = document.querySelectorAll('.cta-button, .cta-button-large, .start-btn');

    ctaButtons.forEach(btn => {
        btn.addEventListener('click', function (e) {
            const href = btn.getAttribute('href');

            // Only trigger loader when going to /index
            if (href === '/index') {
                e.preventDefault();

                showLoading('Preparing your assessment form…', btn);

                setTimeout(() => {
                    window.location.href = '/index';
                }, 2000);
            }
        });
    });
}


function normalizeHomeLinks() {
    // If templates have /base as Home, send user safely to '/'
    const homeLinks = document.querySelectorAll('nav a[href="/base"]');
    homeLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            window.location.href = '/';
        });
    });
}

/* =========================================
   4. TABS (Manual / CSV on Index Page)
   ========================================= */

function openTab(evt, tabName) {
    const tabContents = document.getElementsByClassName('tab-content');
    const tabButtons = document.getElementsByClassName('tab-btn');

    // Hide all tab contents
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
        tabContents[i].style.display = 'none';
    }

    // Deactivate all buttons
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }

    // Show target content
    const target = document.getElementById(tabName);
    if (target) {
        target.style.display = 'block';
        // slight delay to allow CSS transitions
        setTimeout(() => {
            target.classList.add('active');
        }, 10);
    }

    // Activate button if event is provided
    if (evt && evt.currentTarget) {
        evt.currentTarget.classList.add('active');
    } else {
        // If opened programmatically, ensure the first tab button matches
        if (tabButtons.length > 0 && tabName === 'manual') {
            tabButtons[0].classList.add('active');
        }
    }
}

// Make openTab globally accessible for inline onclick
window.openTab = openTab;

/* =========================================
   5. DYNAMIC MEDICINE BLOCKS (Index Page Manual Form)
   ========================================= */

function addMedicine() {
    const container = document.querySelector('.medicines-container');
    const template = document.getElementById('medicine-template');
    if (!container || !template) return;

    // Clone template
    const clone = template.cloneNode(true);

    // Remove id from clone to keep only one template with that ID
    clone.removeAttribute('id');

    // Clear inputs
    const inputs = clone.querySelectorAll('input');
    inputs.forEach(input => {
        if (input.type === 'number') {
            // Set sensible defaults for some fields
            if (input.name === 'units[]') {
                input.value = '1';
            } else if (input.name === 'freq[]') {
                input.value = '2';
            } else if (input.name === 'days[]') {
                input.value = '1';
            } else if (input.name === 'hours[]') {
                input.value = '4';
            } else {
                input.value = '';
            }
        } else {
            input.value = '';
        }
    });

    // Update medicine number
    const existingCards = container.querySelectorAll('.medicine-card');
    const newIndex = existingCards.length + 1;
    const numberSpan = clone.querySelector('.med-num');
    if (numberSpan) {
        numberSpan.textContent = newIndex;
    }

    // Append with small entrance animation
    clone.style.opacity = '0';
    clone.style.transform = 'translateY(10px)';
    container.appendChild(clone);
    requestAnimationFrame(() => {
        clone.style.transition = 'opacity 0.25s ease-out, transform 0.25s ease-out';
        clone.style.opacity = '1';
        clone.style.transform = 'translateY(0)';
    });
}

function removeMedicine(btn) {
    const card = btn.closest('.medicine-card');
    const container = document.querySelector('.medicines-container');
    if (!card || !container) return;

    const cards = container.querySelectorAll('.medicine-card');
    if (cards.length <= 1) {
        // Shake button to indicate at least one medicine is required
        btn.style.animation = 'shake 0.4s ease';
        setTimeout(() => {
            btn.style.animation = '';
        }, 400);
        alert('At least one medicine is required.');
        return;
    }

    // Fade out then remove
    card.style.transition = 'opacity 0.25s ease-out, transform 0.25s ease-out';
    card.style.opacity = '0';
    card.style.transform = 'translateX(16px)';
    setTimeout(() => {
        card.remove();
        // Re-number remaining cards
        const remainingCards = container.querySelectorAll('.medicine-card');
        remainingCards.forEach((c, idx) => {
            const num = c.querySelector('.med-num');
            if (num) num.textContent = idx + 1;
        });
    }, 260);
}

// Make add/remove functions available for inline HTML onclick
window.addMedicine = addMedicine;
window.removeMedicine = removeMedicine;

/* =========================================
   6. CSV TEMPLATE DOWNLOAD (Help Page)
   ========================================= */

function downloadTemplate() {
    // Columns aligned with docs text (adjust to your exact feature set if needed)
    const header = [
        'patient_age',
        'weight_kg',
        'sex',
        'is_pregnant',
        'has_liver_disease',
        'has_kidney_disease',
        'has_heart_disease',
        'has_gi_ulcer_or_gastritis',
        'has_asthma_or_copd',
        'has_diabetes',
        'num_medicines',
        'total_daily_dose_mg',
        'liver_load_daily',
        'kidney_load_daily',
        'heart_load_daily',
        'gi_load_daily',
        'lungs_load_daily',
        'effective_liver_load',
        'effective_kidney_load'
    ];

    const row1 = [
        '45',
        '70.5',
        'male',
        '0',
        '0',
        '0',
        '1',
        '0',
        '0',
        '1',
        '3',
        '1200.0',
        '1.8',
        '0.7',
        '1.2',
        '0.3',
        '0.4',
        '2.4',
        '1.1'
    ];

    const row2 = [
        '32',
        '60.0',
        'female',
        '1',
        '0',
        '0',
        '0',
        '0',
        '0',
        '0',
        '1',
        '500.0',
        '0.9',
        '0.2',
        '0.4',
        '0.1',
        '0.1',
        '1.0',
        '0.5'
    ];

    const csvLines = [
        header.join(','),
        row1.join(','),
        row2.join(',')
    ];

    const csvContent = 'data:text/csv;charset=utf-8,' + csvLines.join('\n');
    const encodedUri = encodeURI(csvContent);

    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'ToxiDose_Template.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

window.downloadTemplate = downloadTemplate;

/* =========================================
   7. FILE INPUT UI (Show selected CSV name)
   ========================================= */

function enhanceFileInput() {
    const fileInput = document.getElementById('file-input');
    if (!fileInput) return;

    const uploadText = document.querySelector('.upload-text');

    fileInput.addEventListener('change', function (e) {
        const file = e.target.files && e.target.files[0];
        if (file && uploadText) {
            uploadText.textContent = file.name;
        } else if (uploadText) {
            uploadText.textContent = 'Drag & drop your CSV file here or click to browse';
        }
    });
}

/* =========================================
   8. FORM SUBMIT LOADING (Analyze Risk)
   ========================================= */

function setupFormLoading() {
    // Any form that has a .submit-btn will get the 2s loading overlay
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const submitBtn = form.querySelector('.submit-btn');
        if (!submitBtn) return;

        form.addEventListener('submit', function (e) {
            // Prevent immediate submit, show overlay, then submit
            e.preventDefault();

            showLoading('Analyzing toxicity risk…', submitBtn);

            setTimeout(() => {
                form.submit();
            }, 2000);
        });
    });
}

/* =========================================
   9. FAQ Toggle (Help page)
   ========================================= */

function setupFAQToggle() {
    const faqItems = document.querySelectorAll('.faq-item');
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        if (!question || !answer) return;

        answer.style.display = 'none';

        question.addEventListener('click', () => {
            const isOpen = answer.style.display === 'block';
            answer.style.display = isOpen ? 'none' : 'block';
        });
    });
}

/* =========================================
   10. EXTRA: Add keyframes for shake (for remove button)
   ========================================= */

(function injectShakeKeyframes() {
    const styleSheet = document.createElement('style');
    styleSheet.innerText = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-4px); }
        75% { transform: translateX(4px); }
    }`;
    document.head.appendChild(styleSheet);
})();

// =============================
// Tablet Overlay for Top-Right Nav Links
// =============================
document.addEventListener("DOMContentLoaded", function () {
    const navLinks = document.querySelectorAll(".nav-link-tablet");

    navLinks.forEach((link) => {
        const imgPath = link.getAttribute("data-tablet-img");
        if (imgPath) {
            link.style.setProperty("--tablet-img-url", `url("${imgPath}")`);
        }
        // no need to toggle .active here – Flask/Jinja handles it per page
    });
});

