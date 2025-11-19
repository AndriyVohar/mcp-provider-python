"""Розумний клієнт, де Ollama сама управляє MCP інструментами."""

import asyncio
import json
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx


class SmartChatClient:
    """Клієнт з підтримкою tool use - модель сама вирішує, які інструменти використовувати."""

    def __init__(self, model: str = "llama2"):
        self.model = model
        self.session = None
        self.available_tools = []

    async def initialize(self, session: ClientSession):
        """Ініціалізація та отримання списку доступних інструментів."""
        self.session = session
        # Отримуємо список всіх доступних інструментів від MCP серверу
        tools_result = await session.list_tools()
        self.available_tools = tools_result.tools
        print(f"✅ Завантажено {len(self.available_tools)} інструментів від MCP сервера")

    async def query_ollama_with_tools(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Запитує Ollama з можливістю використання інструментів.
        Модель сама вирішує, які інструменти їй потрібні.

        Args:
            user_message: Повідомлення від користувача
            max_iterations: Максимальна кількість ітерацій (щоб уникнути нескінченних циклів)

        Returns:
            Фінальна відповідь від моделі
        """

        # Формуємо опис доступних інструментів
        tools_description = self._format_tools_description()

        # Системний промпт з інструкціями про використання інструментів
        system_prompt = f"""Ти дружелюбний помічник, який може використовувати наступні інструменти:

{tools_description}

ІНСТРУКЦІЇ:
1. Аналізуй запит користувача
2. Якщо тобі потрібна інформація, яку ти можеш отримати через інструменти, використовуй їх
3. ⚠️ ВАЖЛИВО: При використанні інструменту search_duckduckgo, ЗАВЖДИ перекладай запит на АНГЛІЙСЬКУ мову перед пошуком. DuckDuckGo краще працює з англійськими запитами!
   Приклад: якщо користувач запитає "яка погода в Київі?", ти маєш шукати "weather in Kyiv today"
4. Щоб викликати інструмент, напиши JSON в цьому форматі:
   {{
     "tool": "назва_інструменту",
     "args": {{"параметр1": "значення1", "параметр2": "значення2"}}
   }}
5. Після отримання результату від інструменту, використай цю інформацію для відповіді
6. Завжди відповідай українською мовою користувачу
7. Будь дружелюбним та корисним
"""

        conversation_history = []

        for iteration in range(max_iterations):
            print(f"\n[Ітерація {iteration + 1}]")

            # Будуємо повідомлення для моделі
            if iteration == 0:
                # Перший виклик
                full_message = user_message
                conversation_history.append({"role": "user", "content": user_message})
            else:
                # Додаємо контекст попередніх кроків
                full_message = f"{user_message}\n\n[Контекст попередніх кроків]\n"
                for msg in conversation_history[-4:]:  # Останні 4 повідомлення для контексту
                    full_message += f"{msg['role']}: {msg['content'][:200]}...\n"

            # Запитуємо Ollama
            try:
                response = await self._call_ollama(system_prompt, full_message)
            except Exception as e:
                return f"❌ Помилка при звернення до Ollama: {e}"

            # Перевіряємо, чи в відповіді є виклик інструменту
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # Модель не вимагає інструментів, це фінальна відповідь
                print(f"\n🤖 Ollama дала фінальну відповідь (без використання інструментів)")
                return response

            # Виконуємо інструменти та збираємо результати
            tool_results = []
            for tool_call in tool_calls:
                print(f"🔧 Використовую інструмент: {tool_call['tool']}({tool_call['args']})")

                try:
                    result = await self.session.call_tool(
                        tool_call['tool'],
                        tool_call['args']
                    )
                    tool_result = result.content[0].text
                    print(f"✅ Результат: {tool_result[:100]}...")
                except Exception as e:
                    tool_result = f"Помилка при виконанні інструменту: {e}"
                    print(f"❌ {tool_result}")

                tool_results.append({
                    "tool": tool_call['tool'],
                    "result": tool_result
                })

            # Добавляємо відповідь моделі і результати в історію
            conversation_history.append({"role": "assistant", "content": response})

            # Формуємо повідомлення про результати для наступного запиту
            results_message = "Ось результати виконання інструментів:\n"
            for tr in tool_results:
                results_message += f"\n📌 Інструмент '{tr['tool']}':\n{tr['result']}\n"

            results_message += "\nТепер, используючи цю інформацію, дай остаточну відповідь користувачу."

            conversation_history.append({"role": "user", "content": results_message})
            user_message = results_message

            # Якщо це остання ітерація, попросимо фінальну відповідь
            if iteration == max_iterations - 1:
                print(f"\n[Ітерація {iteration + 2}] - ОСТАТОЧНА")
                try:
                    final_response = await self._call_ollama(system_prompt, user_message)
                    return final_response
                except Exception as e:
                    return f"❌ Помилка при отриманні остаточної відповіді: {e}"

    async def _call_ollama(self, system_prompt: str, prompt: str) -> str:
        """Запитує Ollama API."""
        try:
            url = "http://localhost:11434/api/generate"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "Помилка: порожня відповідь від моделі")

        except httpx.ConnectError:
            raise Exception("Ollama недоступна на http://localhost:11434")
        except Exception as e:
            raise Exception(f"Помилка при запиті до Ollama: {e}")

    def _format_tools_description(self) -> str:
        """Форматує опис доступних інструментів."""
        description = ""
        for tool in self.available_tools:
            description += f"\n📌 {tool.name}\n"
            description += f"   Опис: {tool.description}\n"
            if tool.inputSchema:
                # Форматуємо параметри
                schema = tool.inputSchema
                if hasattr(schema, 'properties'):
                    description += "   Параметри:\n"
                    for param_name, param_info in schema.properties.items():
                        param_type = param_info.get('type', 'unknown') if isinstance(param_info, dict) else 'unknown'
                        description += f"     - {param_name} ({param_type})\n"

        return description

    def _extract_tool_calls(self, response: str) -> list:
        """Витягує виклики інструментів з відповіді моделі."""
        tool_calls = []

        # Різні формати викликів інструментів:
        # 1. {"tool": "name", "args": {...}}
        # 2. {"tool": "name", "args": {}}
        # 3. {"tool": "name"}
        # 4. {tool: "name", args: {...}} (без лапок)

        # Спробуємо знайти всі JSON об'єкти в тексті
        # Шукаємо відкриту фігурну дужку до закритої
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

        potential_jsons = re.finditer(json_pattern, response, re.DOTALL)

        for match in potential_jsons:
            json_str = match.group(0)

            try:
                # Спробуємо спарсити як JSON (змінюємо одинарні лапки на подвійні)
                corrected_json = json_str.replace("'", '"')
                data = json.loads(corrected_json)

                # Перевіряємо, чи це об'єкт з полем "tool"
                if isinstance(data, dict) and "tool" in data:
                    tool_name = data.get("tool")
                    args = data.get("args", {})

                    # Переконуємось, що args - це словник
                    if not isinstance(args, dict):
                        args = {}

                    tool_calls.append({
                        "tool": tool_name,
                        "args": args
                    })
                    print(f"  → Знайден виклик інструменту: {tool_name}({args})")
            except (json.JSONDecodeError, ValueError, TypeError):
                # Цей JSON не містить інструмент, пропускаємо
                pass

        return tool_calls


async def main():
    """Основна функція."""

    print("🚀 Розумний чат клієнт з інструментами")
    print("=" * 70)

    # Параметри для запуску MCP серверу
    server_params = StdioServerParameters(
        command="python3",
        args=["mcp_server.py"],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Ініціалізація сесії
                await session.initialize()

                # Запитуємо назву моделі
                print("\n📋 Доступні моделі Ollama: llama2, mistral, neural-chat, yi та інші")
                model_name = input("🎯 Яку модель використовувати? (за замовчуванням 'gemma3:4b'): ").strip()
                if not model_name:
                    model_name = "gemma3:4b"

                # Ініціалізуємо клієнт
                client = SmartChatClient(model=model_name)
                await client.initialize(session)

                print(f"\n✅ Готово! Тепер модель {model_name} може сама викликати інструменти.")
                print("=" * 70)

                # Інтерактивний цикл
                while True:
                    print("\n" + "=" * 70)
                    user_input = input("💬 Ви (або 'exit' для виходу): ").strip()

                    if user_input.lower() == 'exit':
                        print("До побачення! 👋")
                        break

                    if not user_input:
                        print("⚠️  Будь ласка, напишіть щось.")
                        continue

                    print("\n🤔 Ollama обробляє ваше повідомлення...\n")

                    response = await client.query_ollama_with_tools(user_input)

                    print("\n" + "=" * 70)
                    print("🤖 Остаточна відповідь Ollama:")
                    print("-" * 70)
                    print(response)
                    print("-" * 70)

    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
