# API Troubleshooter

## 1. Missing API key

Error example:

ANTHROPIC_API_KEY is not set

Meaning:

The program cannot find the API key.

Fix:

Create a .env file and add:

ANTHROPIC_API_KEY=your_api_key_here

Do not commit .env to GitHub.

---

## 2. Wrong API key

Error example:

authentication_error

Meaning:

The API key is invalid, expired, or copied incorrectly.

Fix:

Generate a new key from the provider console and update the .env file.

---

## 3. Rate limit

Error example:

rate_limit_error

Meaning:

Too many requests were sent in a short time.

Fix:

Wait for a while or reduce request frequency.

---

## 4. Insufficient credits

Error example:

billing_error

Meaning:

The account does not have enough API credits.

Fix:

Add credits or use another provider.

---

## 5. Network error

Error example:

ConnectionError or TimeoutError

Meaning:

The computer cannot connect to the API server.

Fix:

Check the network, proxy, VPN, or firewall settings.
