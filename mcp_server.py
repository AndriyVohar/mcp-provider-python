"""MCP Сервер для Ollama з прикладами інструментів."""

from mcp.server.fastmcp import FastMCP
import datetime
from pathlib import Path
import httpx

# Ініціалізація MCP серверу
mcp = FastMCP("my-ollama-tools")


@mcp.tool()
def get_current_time() -> str:
    """Повертає поточний час."""
    return datetime.datetime.now().strftime("%H:%M:%S")


@mcp.tool(name="sum", description="Обчислює суму двох цілих чисел. Параметри: a (int), b (int)")
def calculate_sum(a: int, b: int) -> int:
    """Обчислює суму двох цілих чисел."""
    return a + b


@mcp.tool(name="get_date", description="Повертає поточну дату у форматі YYYY-MM-DD")
def get_current_date() -> str:
    """Повертає поточну дату у форматі YYYY-MM-DD."""
    return datetime.datetime.now().strftime("%Y-%m-%d")


@mcp.tool(name="multiply", description="Множить два числа. Параметри: a (float), b (float)")
def multiply_numbers(a: float, b: float) -> float:
    """Множить два числа."""
    return a * b


@mcp.tool(name="read_notes", description="Читає вміст файлу notes.txt з кореня проекту")
def read_notes() -> str:
    """Читає вміст `notes.txt` з кореня проекту.

    Повертає повідомлення про помилку, якщо файл не знайдено або не вдається прочитати.
    Якщо файл дуже великий, повертається перші 10000 символів з позначкою про обрізання.
    """
    path = Path("notes.txt")
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "notes.txt not found."
    except Exception as e:
        return f"Error reading notes.txt: {e}"
    max_len = 10000
    if len(content) > max_len:
        return content[:max_len] + "\n\n...[truncated]"
    return content


@mcp.tool(name="search_duckduckgo", description="Пошук інформації в інтернеті через DuckDuckGo API. Параметри: query (str), max_results (int, за замовчуванням 5)")
def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Пошук інформації в інтернеті через DuckDuckGo API.

    Args:
        query: Запит для пошуку
        max_results: Максимальна кількість результатів (за замовчуванням 5)

    Повертає відформатовані результати пошуку або повідомлення про помилку.
    """
    try:
        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }

        with httpx.Client() as client:
            response = client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

        results = []

        # Додаємо основну відповідь (Instant Answer)
        if data.get("AbstractText"):
            results.append(f"📌 {data.get('AbstractTitle', 'Result')}: {data['AbstractText']}")

        # Додаємо перші N результатів з Related Topics
        if data.get("RelatedTopics"):
            related = data["RelatedTopics"][:max_results]
            for item in related:
                if isinstance(item, dict):
                    text = item.get("Text", "")
                    url_item = item.get("FirstURL", "")
                    if text:
                        results.append(f"🔗 {text}\n   URL: {url_item}")

        if not results:
            return f"Жодних результатів не знайдено для запиту: {query}"

        return "\n\n".join(results)

    except httpx.TimeoutException:
        return "Помилка: Час очікування запиту вийшов. Спробуйте ще раз."
    except httpx.RequestError as e:
        return f"Помилка мережі при запиті до DuckDuckGo: {e}"
    except Exception as e:
        return f"Помилка при пошуку: {e}"


if __name__ == "__main__":
    # Запуск серверу через stdio транспорт
    mcp.run(transport='stdio')
