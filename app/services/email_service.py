"""
Email Service — inbound parsing and outbound sending.
Supports SendGrid / Mailgun. Falls back to console logging in dev mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InboundEmail:
    """Parsed inbound email from webhook payload."""
    message_id: str
    sender: str
    sender_name: str
    recipient: str
    subject: str
    body_plain: str
    body_html: Optional[str] = None
    attachments: List[Dict] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_sendgrid_webhook(cls, payload: Dict[str, Any]) -> "InboundEmail":
        """Parse a SendGrid Inbound Parse webhook payload."""
        return cls(
            message_id=payload.get("headers", "").split("Message-ID:")[-1].split("\n")[0].strip()
                       if "Message-ID:" in payload.get("headers", "") else "unknown",
            sender=payload.get("from", ""),
            sender_name=payload.get("from", "").split("<")[0].strip(),
            recipient=payload.get("to", ""),
            subject=payload.get("subject", "(no subject)"),
            body_plain=payload.get("text", ""),
            body_html=payload.get("html"),
        )

    @classmethod
    def from_mailgun_webhook(cls, payload: Dict[str, Any]) -> "InboundEmail":
        """Parse a Mailgun inbound webhook payload."""
        return cls(
            message_id=payload.get("Message-Id", "unknown"),
            sender=payload.get("sender", ""),
            sender_name=payload.get("From", "").split("<")[0].strip(),
            recipient=payload.get("recipient", ""),
            subject=payload.get("subject", "(no subject)"),
            body_plain=payload.get("body-plain", ""),
            body_html=payload.get("body-html"),
        )


class EmailService:
    def __init__(
        self,
        sendgrid_api_key: Optional[str] = None,
        mailgun_api_key: Optional[str] = None,
        mailgun_domain: Optional[str] = None,
        from_email: str = "support@yourdomain.com",
        from_name: str = "Support Team",
    ):
        self.sendgrid_key = sendgrid_api_key
        self.mailgun_key = mailgun_api_key
        self.mailgun_domain = mailgun_domain
        self.from_email = from_email
        self.from_name = from_name
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if self.sendgrid_key:
            return "sendgrid"
        if self.mailgun_key:
            return "mailgun"
        return "console"

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> bool:
        """Send an outbound email. Returns True on success."""
        if self._provider == "sendgrid":
            return await self._sendgrid_send(to, subject, body, html_body, reply_to)
        if self._provider == "mailgun":
            return await self._mailgun_send(to, subject, body, html_body, reply_to)
        return self._console_send(to, subject, body)

    async def send_ticket_confirmation(
        self,
        to: str,
        customer_name: str,
        ticket_id: str,
        subject: str,
    ) -> bool:
        email_subject = f"[{ticket_id}] We've received your request"
        body = (
            f"Hi {customer_name},\n\n"
            f"Thank you for contacting us. We've created a support ticket to track your issue:\n\n"
            f"  Ticket ID: {ticket_id}\n"
            f"  Subject: {subject}\n\n"
            f"Our team will get back to you within 24 hours.\n\n"
            f"Best regards,\n{self.from_name}"
        )
        return await self.send(to, email_subject, body)

    async def send_escalation_notification(
        self,
        agent_email: str,
        ticket_id: str,
        customer_email: str,
        summary: str,
        urgency: str,
    ) -> bool:
        subject = f"[{urgency.upper()}] Escalation: {ticket_id}"
        body = (
            f"A conversation has been escalated and requires your attention.\n\n"
            f"Ticket: {ticket_id}\n"
            f"Customer: {customer_email}\n"
            f"Urgency: {urgency}\n\n"
            f"Summary:\n{summary}\n\n"
            f"Please log in to the support dashboard to respond."
        )
        return await self.send(agent_email, subject, body)

    # ------------------------------------------------------------------
    # SendGrid
    # ------------------------------------------------------------------

    async def _sendgrid_send(
        self, to: str, subject: str, body: str,
        html_body: Optional[str], reply_to: Optional[str]
    ) -> bool:
        try:
            import httpx
            payload = {
                "personalizations": [{"to": [{"email": to}]}],
                "from": {"email": self.from_email, "name": self.from_name},
                "subject": subject,
                "content": [
                    {"type": "text/plain", "value": body},
                    *(
                        [{"type": "text/html", "value": html_body}]
                        if html_body else []
                    ),
                ],
            }
            if reply_to:
                payload["reply_to"] = {"email": reply_to}

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.sendgrid_key}"},
                )
                resp.raise_for_status()
            logger.info("Email sent via SendGrid to %s", to)
            return True
        except Exception as exc:
            logger.error("SendGrid send failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Mailgun
    # ------------------------------------------------------------------

    async def _mailgun_send(
        self, to: str, subject: str, body: str,
        html_body: Optional[str], reply_to: Optional[str]
    ) -> bool:
        try:
            import httpx
            data = {
                "from": f"{self.from_name} <{self.from_email}>",
                "to": to,
                "subject": subject,
                "text": body,
            }
            if html_body:
                data["html"] = html_body
            if reply_to:
                data["h:Reply-To"] = reply_to

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://api.mailgun.net/v3/{self.mailgun_domain}/messages",
                    auth=("api", self.mailgun_key),
                    data=data,
                )
                resp.raise_for_status()
            logger.info("Email sent via Mailgun to %s", to)
            return True
        except Exception as exc:
            logger.error("Mailgun send failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Console (development fallback)
    # ------------------------------------------------------------------

    def _console_send(self, to: str, subject: str, body: str) -> bool:
        print(f"\n{'='*60}")
        print(f"📧 EMAIL (console mode)")
        print(f"To: {to}")
        print(f"Subject: {subject}")
        print(f"Body:\n{body}")
        print(f"{'='*60}\n")
        return True
