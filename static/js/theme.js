/**
 * TicketAI - Global Theme Switcher (Dark / Light Mode)
 * Persists selection in localStorage and responds to user toggle.
 */

(function () {
    const THEME_KEY = 'ticketai-theme';

    // 1. Determine active theme (default to dark if not set)
    function getPreferredTheme() {
        const savedTheme = localStorage.getItem(THEME_KEY);
        if (savedTheme) {
            return savedTheme;
        }
        return 'dark'; // Default SaaS modern dark
    }

    // 2. Apply theme attribute immediately to <html>
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
        updateToggleButtons(theme);
    }

    // 3. Update icon & title on any toggle buttons present on the page
    function updateToggleButtons(theme) {
        const buttons = document.querySelectorAll('.theme-toggle-btn');
        buttons.forEach(btn => {
            const isDark = theme === 'dark';
            btn.innerHTML = isDark 
                ? '<i class="fas fa-sun" style="color: #fbbf24;"></i>' 
                : '<i class="fas fa-moon" style="color: #6366f1;"></i>';
            btn.setAttribute('title', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
            btn.setAttribute('aria-label', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
        });
    }

    // 4. Toggle theme function exposed globally
    window.toggleTheme = function () {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
    };

    // Apply preferred theme immediately upon script evaluation
    applyTheme(getPreferredTheme());

    // Update buttons once DOM is fully parsed
    document.addEventListener('DOMContentLoaded', function () {
        updateToggleButtons(document.documentElement.getAttribute('data-theme') || 'dark');
    });
})();
