"""
dataset.py
Manually curated labeled dataset for training the classifier.
Each entry is a (message, label) tuple.
Labels: Complaint, Feedback, Inquiry
"""

DATA = [
    # ── Complaints ──────────────────────────────────────────────────────────
    ("Your service has been down for two days and I'm losing money because of it.", "Complaint"),
    ("This is absolutely unacceptable. I've been waiting for a refund for 3 weeks!", "Complaint"),
    ("I ordered a product last month and it still hasn't arrived. Very disappointed.", "Complaint"),
    ("Your customer support is terrible. No one picks up the phone.", "Complaint"),
    ("I was charged twice for the same order and nobody is helping me.", "Complaint"),
    ("The app crashes every time I try to open it. This is frustrating.", "Complaint"),
    ("I expected much better quality for the price I paid.", "Complaint"),
    ("My account was locked without any explanation. Fix this immediately.", "Complaint"),
    ("The delivery driver left my package in the rain and everything is ruined.", "Complaint"),
    ("I've submitted three tickets and no one has responded. Completely unacceptable.", "Complaint"),
    ("Your website keeps logging me out. I can't get anything done.", "Complaint"),
    ("The product I received is broken and it looks like it was used before.", "Complaint"),
    ("I've been a loyal customer for years but this experience has been awful.", "Complaint"),
    ("Your chatbot gave me wrong information and now I missed my appointment.", "Complaint"),
    ("The subscription was supposed to be cancelled but you keep charging me.", "Complaint"),
    ("Waited 45 minutes on hold and then got disconnected. Terrible service.", "Complaint"),
    ("My password reset email never arrived. I've tried five times.", "Complaint"),
    ("You sent me the wrong item and now I have to pay for return shipping.", "Complaint"),
    ("The noise from the product is unbearable. It sounds like a machine gun.", "Complaint"),
    ("I'm extremely unhappy with how my complaint was handled last time.", "Complaint"),

    # ── Feedback ────────────────────────────────────────────────────────────
    ("The new dashboard is really clean and easy to navigate. Great work!", "Feedback"),
    ("I love the redesign but the font size feels a bit small on mobile.", "Feedback"),
    ("The onboarding flow is much smoother than it used to be. Good improvement.", "Feedback"),
    ("Your team did an amazing job on the latest update. Keep it up!", "Feedback"),
    ("The dark mode looks great but the sidebar icons are hard to see.", "Feedback"),
    ("Checkout is now super fast. I really appreciate the improvement.", "Feedback"),
    ("The new search feature is excellent. Saves me a lot of time.", "Feedback"),
    ("I think the pricing page could be clearer about what's included in each tier.", "Feedback"),
    ("The mobile app feels much snappier after the last update. Nice work.", "Feedback"),
    ("Your blog content is very helpful and well written. Please post more often.", "Feedback"),
    ("The tutorial videos are a bit too long. Would prefer shorter clips.", "Feedback"),
    ("I really enjoy using the product. The integrations with other tools are seamless.", "Feedback"),
    ("The filters on the report page are a great addition. Very useful.", "Feedback"),
    ("Suggestion: adding keyboard shortcuts would make the workflow much faster.", "Feedback"),
    ("The confirmation emails are helpful but could include more order details.", "Feedback"),
    ("The API documentation is well-structured and easy to follow.", "Feedback"),
    ("I think the color contrast on some buttons could be improved for accessibility.", "Feedback"),
    ("The new notification settings give much better control. Appreciate it.", "Feedback"),
    ("Overall a great product. My only feedback is that the loading time can be slow.", "Feedback"),
    ("The team has clearly put a lot of thought into the user experience. Well done.", "Feedback"),

    # ── Inquiries ───────────────────────────────────────────────────────────
    ("What are the pricing plans for the enterprise tier?", "Inquiry"),
    ("Do you offer a free trial for new users?", "Inquiry"),
    ("How do I reset my password?", "Inquiry"),
    ("Is there an API available for integration with third-party tools?", "Inquiry"),
    ("What countries does your service support?", "Inquiry"),
    ("Can I upgrade my plan in the middle of a billing cycle?", "Inquiry"),
    ("How long does standard shipping typically take?", "Inquiry"),
    ("What payment methods do you accept?", "Inquiry"),
    ("Is my data encrypted and how is it stored?", "Inquiry"),
    ("How do I cancel my subscription?", "Inquiry"),
    ("Can I export my data in CSV format?", "Inquiry"),
    ("Do you have a mobile app for Android?", "Inquiry"),
    ("What is your refund policy?", "Inquiry"),
    ("How many users can I add to a single account?", "Inquiry"),
    ("Do you offer discounts for non-profit organizations?", "Inquiry"),
    ("Is there a way to bulk import contacts?", "Inquiry"),
    ("Where can I find the release notes for the latest version?", "Inquiry"),
    ("What are your support hours?", "Inquiry"),
    ("Can I use the service offline?", "Inquiry"),
    ("Is two-factor authentication available for my account?", "Inquiry"),
]
