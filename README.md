# parkpulse

## web frontend

This project now includes a Flask frontend so you can upload a parking video and get an annotated output video.

> Important: `model.p` was trained with `scikit-learn==1.1.3`. For accurate predictions, run with Python 3.10 and scikit-learn 1.1.3.

### run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended (macOS/Homebrew Python 3.10):

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
pip install "numpy<2" scikit-learn==1.1.3 scikit-image==0.19.3 opencv-python==4.6.0.66 Flask==3.1.0
```

2. Start the web app:

```bash
python web_app.py
```

3. Open:

```text
http://127.0.0.1:5000
```

### stripe checkout setup (real payments in India)

1. Install dependencies (includes Stripe SDK):

```bash
pip install -r requirements.txt
```

2. Set your Stripe secret key (test mode recommended first):

```bash
export STRIPE_SECRET_KEY=sk_test_xxx
```

or create a `.env` file in the project root:

```env
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
```

Set webhook and base URL (recommended for production-like flow):

```bash
export STRIPE_WEBHOOK_SECRET=whsec_xxx
export APP_BASE_URL=https://your-domain.example
```

3. Stripe Dashboard setup for India:

- Set account country to India in Stripe account profile.
- Keep settlement currency in INR for this flow.
- Enable payment methods used in India (Cards, UPI, and others available to your account) in Dashboard.
- Use dynamic payment methods (already enabled in this app by default) instead of hardcoding method types.

4. Start app and choose `Stripe Checkout (Cards + UPI where available)` in the booking payment modal.

5. After payment, Stripe redirects to:

- `/payments/stripe/status?session_id=...` for live confirmation (works for async rails too)
- `/payments/stripe/cancel` when user cancels

The status page polls:

- `GET /api/payments/stripe/status?session_id=...`

until final success or failure, so users can wait on-page while webhook events arrive.

6. For local webhook testing with Stripe CLI:

```bash
stripe listen --forward-to localhost:5000/payments/stripe/webhook
```

7. In production, always expose `/payments/stripe/webhook` over HTTPS and verify webhook signatures (already implemented in this app).

### go live checklist for India

1. In Stripe Dashboard, ensure India account settings and payment methods (UPI/Cards) are enabled.
2. Set environment variables in deployment:
	- `STRIPE_SECRET_KEY`
	- `STRIPE_WEBHOOK_SECRET`
	- `APP_BASE_URL`
3. Add webhook endpoint in Stripe Dashboard:
	- `https://your-domain/payments/stripe/webhook`
4. Subscribe to events:
	- `checkout.session.completed`
	- `checkout.session.async_payment_succeeded`
	- `checkout.session.async_payment_failed`
	- `checkout.session.expired`

### what you can do

- Upload a video file (`.mp4`, `.mov`, `.avi`, `.mkv`)
- Process parking occupancy with the existing model and mask
- Preview and download the annotated output video
- Book available slots from the map
- Confirm bookings through the built-in demo payment flow (`/api/bookings/pay`)
