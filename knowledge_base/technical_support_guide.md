# Technical Support Knowledge Guide & Troubleshooting

## 1. Password Reset & Login Failures
- If a user receives an "Invalid credentials" error:
  1. Direct user to click "Forgot Password" on the login screen.
  2. Enter the account registered email.
  3. Verify password reset email arrival within 10 minutes (check Spam/Junk folder).
  4. Ensure new password has at least 8 characters, one number, and one uppercase letter.
  5. Clear browser cookies and session storage if the page loops.

## 2. Resolving HTTP 500 Internal Server Errors
- Steps for users encountering HTTP 500 Internal Server Error:
  1. Refresh the web page using hard-refresh (`Ctrl + F5` on Windows, `Cmd + Shift + R` on Mac).
  2. Verify that network proxy or VPN is not blocking API endpoints.
  3. Try accessing the dashboard via an Incognito / Private browsing window.
  4. If the error persists, report the exact URL and timestamp to engineering for server log analysis.

## 3. Two-Factor Authentication (2FA) Recovery
- If the user lost access to their authenticator application:
  1. Use one of the 8-digit emergency backup codes generated at setup.
  2. If emergency codes are unavailable, the user must undergo identity verification via phone SMS or registered corporate email.

