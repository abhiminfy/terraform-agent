import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChatMemory:
    """Local file-based chat memory system with title generation and settings support"""

    def __init__(self, storage_dir: str = "chat_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.max_history_length = 50  # Keep last 50 messages

    def _get_chat_file(self, chat_id: str) -> Path:
        """Get the file path for a specific chat"""
        safe_chat_id = "".join(c for c in chat_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"chat_{safe_chat_id}.json"

    def save_message(
        self,
        chat_id: str,
        role: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Save a message to chat history with auto title generation"""
        try:
            chat_file = self._get_chat_file(chat_id)

            # Load existing history
            chat_data = {}
            if chat_file.exists():
                try:
                    with open(chat_file, "r", encoding="utf-8") as f:
                        chat_data = json.load(f)
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"Corrupted chat file for {chat_id}, starting fresh")
                    chat_data = {}

            # Initialize chat data structure
            if not chat_data:
                chat_data = {
                    "chat_id": chat_id,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "message_count": 0,
                    "messages": [],
                    "title": "",
                    "archived": False,
                    "settings": {},
                    "summary": "",
                }

            history = chat_data.get("messages", [])

            # Add new message
            new_message = {
                "role": role,  # 'user' or 'assistant'
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            history.append(new_message)

            # Auto-generate title after first assistant response
            if not chat_data.get("title") and role == "assistant" and len(history) >= 2:
                chat_data["title"] = self._generate_chat_title(history)

            # Generate summary after every few messages
            if len(history) % 6 == 0:  # Update summary every 6 messages
                chat_data["summary"] = self._generate_chat_summary(history)

            # Trim history if too long
            if len(history) > self.max_history_length:
                history = history[-self.max_history_length :]

            # Update chat data
            chat_data.update(
                {
                    "last_updated": datetime.now().isoformat(),
                    "message_count": len(history),
                    "messages": history,
                }
            )

            # Save back to file
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Failed to save message for chat {chat_id}: {str(e)}")
            return False

    def update_chat_metadata(
        self,
        chat_id: str,
        title: str = None,
        archived: bool = None,
        settings: Dict[str, Any] = None,
    ) -> bool:
        """Update chat metadata (title, archived status, settings)"""
        try:
            chat_file = self._get_chat_file(chat_id)

            if not chat_file.exists():
                return False

            with open(chat_file, "r", encoding="utf-8") as f:
                chat_data = json.load(f)

            # Update fields if provided
            if title is not None:
                chat_data["title"] = title

            if archived is not None:
                chat_data["archived"] = archived

            if settings is not None:
                current_settings = chat_data.get("settings", {})
                current_settings.update(settings)
                chat_data["settings"] = current_settings

            chat_data["last_updated"] = datetime.now().isoformat()

            # Save updated data
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Failed to update metadata for chat {chat_id}: {str(e)}")
            return False

    def get_chat_settings(self, chat_id: str) -> Dict[str, Any]:
        """Get settings for a specific chat"""
        try:
            chat_file = self._get_chat_file(chat_id)

            if not chat_file.exists():
                return {}

            with open(chat_file, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
                return chat_data.get("settings", {})

        except Exception as e:
            logger.error(f"Failed to get settings for chat {chat_id}: {str(e)}")
            return {}

    def _generate_chat_title(self, history: List[Dict[str, Any]]) -> str:
        """Generate a title based on the first user message and assistant response"""
        if len(history) < 2:
            return "New Chat"

        try:
            first_user_message = ""
            first_assistant_response = ""

            for msg in history:
                if msg.get("role") == "user" and not first_user_message:
                    first_user_message = msg.get("content", "")
                elif msg.get("role") == "assistant" and not first_assistant_response:
                    first_assistant_response = msg.get("content", "")
                    break

            # Extract key topics from the conversation
            title = self._extract_title_from_content(first_user_message, first_assistant_response)
            return title

        except Exception as e:
            logger.error(f"Failed to generate chat title: {str(e)}")
            return "New Chat"

    def _extract_title_from_content(self, user_msg: str, assistant_msg: str) -> str:
        """Extract a meaningful title from conversation content"""
        # Keywords that indicate infrastructure topics
        infra_keywords = {
            "ec2": "EC2 Instance",
            "s3": "S3 Bucket",
            "rds": "RDS Database",
            "vpc": "VPC Setup",
            "lambda": "Lambda Function",
            "load balancer": "Load Balancer",
            "security group": "Security Group",
            "terraform": "Terraform Code",
            "infrastructure": "Infrastructure",
            "deploy": "Deployment",
            "cloud": "Cloud Setup",
        }

        combined_text = (user_msg + " " + assistant_msg).lower()

        # Find matching keywords
        for keyword, title in infra_keywords.items():
            if keyword in combined_text:
                return title

        # Extract first few words from user message
        user_words = user_msg.split()[:5]  # First 5 words
        if len(user_words) >= 2:
            title = " ".join(user_words)
            # Capitalize first letter
            title = title[0].upper() + title[1:] if title else "New Chat"
            return title[:50]  # Limit to 50 characters

        return "New Chat"

    def _generate_chat_summary(self, history: List[Dict[str, Any]]) -> str:
        """Generate a summary of the chat conversation"""
        try:
            if len(history) < 3:
                return ""

            # Extract key points from the conversation
            topics = []
            actions = []

            for msg in history[-6:]:  # Look at last 6 messages
                content = msg.get("content", "").lower()

                # Identify topics
                if "terraform" in content or "infrastructure" in content:
                    topics.append("infrastructure")
                if any(service in content for service in ["ec2", "s3", "rds", "vpc", "lambda"]):
                    topics.append("aws_services")
                if "cost" in content or "pricing" in content:
                    topics.append("cost_analysis")
                if "security" in content:
                    topics.append("security")

                # Identify actions
                if msg.get("role") == "assistant":
                    metadata = msg.get("metadata", {})
                    if metadata.get("type") == "terraform":
                        actions.append("generated_terraform")
                    elif "clarify" in content:
                        actions.append("requested_clarification")

            # Create summary
            summary_parts = []
            if topics:
                unique_topics = list(set(topics))
                summary_parts.append(f"Topics: {', '.join(unique_topics)}")

            if actions:
                unique_actions = list(set(actions))
                summary_parts.append(f"Actions: {', '.join(unique_actions)}")

            return " | ".join(summary_parts) if summary_parts else "General conversation"

        except Exception as e:
            logger.error(f"Failed to generate chat summary: {str(e)}")
            return ""

    def get_chat_history(self, chat_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get chat history for a specific chat"""
        try:
            chat_file = self._get_chat_file(chat_id)

            if not chat_file.exists():
                return []

            with open(chat_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                messages = data.get("messages", [])

                # Return last N messages
                return messages[-limit:] if limit else messages

        except Exception as e:
            logger.error(f"Failed to load chat history for {chat_id}: {str(e)}")
            return []

    def get_context_for_prompt(self, chat_id: str, context_length: int = 5) -> str:
        """Get recent chat context formatted for prompts with settings awareness"""
        try:
            history = self.get_chat_history(chat_id, limit=context_length)
            settings = self.get_chat_settings(chat_id)

            if not history:
                return "No previous conversation context."

            context_lines = []

            # Add settings context if available
            if settings:
                settings_context = []
                if settings.get("cloud"):
                    settings_context.append(f"Cloud: {settings['cloud']}")
                if settings.get("region"):
                    settings_context.append(f"Region: {settings['region']}")
                if settings.get("environment"):
                    settings_context.append(f"Environment: {settings['environment']}")
                if settings.get("budget"):
                    settings_context.append(f"Budget: ${settings['budget']}")

                if settings_context:
                    context_lines.append(f"Chat Settings: {', '.join(settings_context)}")
                    context_lines.append("")

            # Add conversation history
            for msg in history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."

                if role == "user":
                    context_lines.append(f"User: {content}")
                elif role == "assistant":
                    context_lines.append(f"Assistant: {content}")

            return "\n".join(context_lines)

        except Exception as e:
            logger.error(f"Failed to get context for {chat_id}: {str(e)}")
            return "Error loading conversation context."

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and its history"""
        try:
            chat_file = self._get_chat_file(chat_id)
            if chat_file.exists():
                chat_file.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete chat {chat_id}: {str(e)}")
            return False

    def list_chats(self) -> List[Dict[str, Any]]:
        """List all available chats with enhanced metadata"""
        try:
            chats = []
            for chat_file in self.storage_dir.glob("chat_*.json"):
                try:
                    with open(chat_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Get preview from last message or summary
                        preview = ""
                        if data.get("summary"):
                            preview = data["summary"]
                        elif data.get("messages"):
                            last_msg = data["messages"][-1]
                            preview = last_msg.get("content", "")[:100]

                        chat_info = {
                            "chat_id": data.get("chat_id"),
                            "title": data.get("title", ""),
                            "last_updated": data.get("last_updated"),
                            "message_count": data.get("message_count", 0),
                            "preview": preview,
                            "archived": data.get("archived", False),
                            "summary": data.get("summary", ""),
                            "settings": data.get("settings", {}),
                            "created_at": data.get("created_at"),
                        }

                        chats.append(chat_info)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping corrupted chat file: {chat_file} - {str(e)}")
                    continue

            # Sort by last updated (most recent first)
            chats.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            return chats

        except Exception as e:
            logger.error(f"Failed to list chats: {str(e)}")
            return []

    def search_chats(self, query: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Search chats by content, title, or summary"""
        try:
            all_chats = self.list_chats()
            matching_chats = []

            query_lower = query.lower()

            for chat in all_chats:
                if not include_archived and chat.get("archived", False):
                    continue

                # Search in title, summary, and preview
                searchable_text = " ".join(
                    [
                        chat.get("title", ""),
                        chat.get("summary", ""),
                        chat.get("preview", ""),
                    ]
                ).lower()

                if query_lower in searchable_text:
                    # Calculate relevance score
                    score = 0
                    if query_lower in chat.get("title", "").lower():
                        score += 3
                    if query_lower in chat.get("summary", "").lower():
                        score += 2
                    if query_lower in chat.get("preview", "").lower():
                        score += 1

                    chat["relevance_score"] = score
                    matching_chats.append(chat)

            # Sort by relevance score
            matching_chats.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return matching_chats

        except Exception as e:
            logger.error(f"Failed to search chats: {str(e)}")
            return []

    def get_chat_stats(self) -> Dict[str, Any]:
        """Get statistics about all chats"""
        try:
            all_chats = self.list_chats()

            total_chats = len(all_chats)
            archived_chats = len([c for c in all_chats if c.get("archived", False)])
            active_chats = total_chats - archived_chats

            total_messages = sum(chat.get("message_count", 0) for chat in all_chats)

            # Find most active chat
            most_active = (
                max(all_chats, key=lambda x: x.get("message_count", 0)) if all_chats else None
            )

            # Calculate average messages per chat
            avg_messages = total_messages / total_chats if total_chats > 0 else 0

            # Find recent activity
            recent_chats = [c for c in all_chats if c.get("last_updated", "")]
            if recent_chats:
                most_recent = max(recent_chats, key=lambda x: x.get("last_updated", ""))
            else:
                most_recent = None

            return {
                "total_chats": total_chats,
                "active_chats": active_chats,
                "archived_chats": archived_chats,
                "total_messages": total_messages,
                "average_messages_per_chat": round(avg_messages, 1),
                "most_active_chat": (
                    {
                        "chat_id": most_active.get("chat_id"),
                        "title": most_active.get("title", ""),
                        "message_count": most_active.get("message_count", 0),
                    }
                    if most_active
                    else None
                ),
                "most_recent_chat": (
                    {
                        "chat_id": most_recent.get("chat_id"),
                        "title": most_recent.get("title", ""),
                        "last_updated": most_recent.get("last_updated"),
                    }
                    if most_recent
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get chat stats: {str(e)}")
            return {
                "error": str(e),
                "total_chats": 0,
                "active_chats": 0,
                "archived_chats": 0,
            }

    def export_chat(self, chat_id: str, format: str = "json") -> Optional[Dict[str, Any]]:
        """Export chat data in specified format"""
        try:
            chat_file = self._get_chat_file(chat_id)

            if not chat_file.exists():
                return None

            with open(chat_file, "r", encoding="utf-8") as f:
                chat_data = json.load(f)

            if format.lower() == "json":
                return chat_data
            elif format.lower() == "markdown":
                return self._export_to_markdown(chat_data)
            else:
                return chat_data  # Default to JSON

        except Exception as e:
            logger.error(f"Failed to export chat {chat_id}: {str(e)}")
            return None

    def _export_to_markdown(self, chat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chat data to markdown format"""
        try:
            md_lines = []

            # Header
            title = chat_data.get("title", "Chat Export")
            md_lines.append(f"# {title}\n")

            # Metadata
            md_lines.append("## Chat Information")
            md_lines.append(f"- **Chat ID**: {chat_data.get('chat_id', 'Unknown')}")
            md_lines.append(f"- **Created**: {chat_data.get('created_at', 'Unknown')}")
            md_lines.append(f"- **Last Updated**: {chat_data.get('last_updated', 'Unknown')}")
            md_lines.append(f"- **Message Count**: {chat_data.get('message_count', 0)}")

            if chat_data.get("summary"):
                md_lines.append(f"- **Summary**: {chat_data.get('summary')}")

            md_lines.append("\n## Conversation\n")

            # Messages
            messages = chat_data.get("messages", [])
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown").title()
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")

                md_lines.append(f"### {role} - Message {i}")
                if timestamp:
                    md_lines.append(f"*{timestamp}*\n")

                # Handle code blocks in content
                if "```" in content:
                    md_lines.append(content)
                else:
                    md_lines.append(content)

                md_lines.append("")  # Empty line between messages

            markdown_content = "\n".join(md_lines)

            return {
                "format": "markdown",
                "content": markdown_content,
                "filename": f"{chat_data.get('chat_id', 'chat')}.md",
            }

        except Exception as e:
            logger.error(f"Failed to convert to markdown: {str(e)}")
            return {"error": str(e)}

    def cleanup_old_chats(self, days_old: int = 30, archive_instead: bool = True) -> Dict[str, Any]:
        """Clean up old chats by archiving or deleting"""
        try:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=days_old)
            processed_count = 0
            archived_count = 0
            deleted_count = 0

            for chat_file in self.storage_dir.glob("chat_*.json"):
                try:
                    with open(chat_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    last_updated = datetime.fromisoformat(data.get("last_updated", "1970-01-01"))

                    if last_updated < cutoff_date:
                        if archive_instead and not data.get("archived", False):
                            # Archive the chat
                            data["archived"] = True
                            data["archived_at"] = datetime.now().isoformat()

                            with open(chat_file, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)

                            archived_count += 1
                        else:
                            # Delete the chat
                            chat_file.unlink()
                            deleted_count += 1

                        processed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing {chat_file} for cleanup: {str(e)}")
                    continue

            logger.info(
                f"Cleanup completed: {processed_count} chats processed, {archived_count} archived, {deleted_count} deleted"
            )

            return {
                "processed_count": processed_count,
                "archived_count": archived_count,
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to cleanup old chats: {str(e)}")
            return {
                "error": str(e),
                "processed_count": 0,
                "archived_count": 0,
                "deleted_count": 0,
            }
