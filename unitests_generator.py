import os
import re
import json
import ast
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from langchain_sambanova import ChatSambaNovaCloud
from langgraph.graph import StateGraph, END
import demjson3


class CodeParser:
    """Parse Python code to extract structure and documentation."""

    @staticmethod
    def extract_function_info(code: str) -> Dict[str, Any]:
        """Extract function information using AST."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function name
                    func_name = node.name

                    # Extract parameters
                    params = []
                    for arg in node.args.args:
                        param_name = arg.arg
                        param_type = None
                        if arg.annotation:
                            param_type = ast.unparse(arg.annotation)
                        params.append({"name": param_name, "type": param_type})

                    # Extract return type
                    return_type = None
                    if node.returns:
                        return_type = ast.unparse(node.returns)

                    # Extract docstring
                    docstring = ast.get_docstring(node)

                    dependencies = CodeParser.extract_dependencies(node)

                    return {
                        "name": func_name,
                        "params": params,
                        "return_type": return_type,
                        "docstring": docstring,
                        "dependencies": dependencies,
                    }
        except SyntaxError as e:
            return {"error": f"Syntax error in code: {str(e)}"}

        return {"error": "No function definition found"}

    @staticmethod
    def extract_dependencies(node: ast.AST) -> List[str]:
        """Extract potential external dependencies from function."""
        dependencies = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if hasattr(subnode.func, "id"):
                    dependencies.append(subnode.func.id)
                elif hasattr(subnode.func, "attr") and hasattr(
                    subnode.func.value, "id"
                ):
                    dependencies.append(
                        f"{subnode.func.value.id}.{subnode.func.attr}"
                    )
        return dependencies

    @staticmethod
    def extract_module_name(file_path: str) -> Optional[str]:
        """Extract module name from file path for import statements."""
        if not file_path:
            return None

        module_name = os.path.basename(file_path)
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        return module_name


class TestOutputFormatter:
    """Format and save generated tests."""

    @staticmethod
    def format_test_file(
        function_name: str,
        test_code: str,
        module_name: Optional[str] = None,
        module_path: Optional[str] = None,
        imports: Optional[List[str]] = None,
    ) -> str:
        """Format a complete test file with proper imports and structure."""
        if imports is None:
            imports = ["pytest"]

        import_section = "\n".join([f"import {imp}" for imp in imports])

        function_import = ""
        if module_name:
            function_import = f"from {module_name} import {function_name}"
        elif module_path:
            rel_path = os.path.relpath(module_path)
            if rel_path.endswith(".py"):
                rel_path = rel_path[:-3]
            rel_path = rel_path.replace("/", ".")
            function_import = f"from {rel_path} import {function_name}"

        if function_import:
            import_section += f"\n{function_import}"

        return f"""# Test file for {function_name}
{import_section}

{test_code}
"""

    @staticmethod
    def save_test_file(
        test_code: str,
        output_path: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> str:
        """Save test code to a file and return the file path."""
        if output_path is None:
            if function_name:
                output_path = f"test_{function_name}.py"
            else:
                output_path = "generated_test.py"

        with open(output_path, "w") as f:
            f.write(test_code)

        return output_path

    @staticmethod
    def format_code(code: str, use_black: bool = True) -> str:
        """Format code using black if available."""
        formatted_code = code

        if use_black:
            try:
                import black

                mode = black.Mode()
                formatted_code = black.format_str(formatted_code, mode=mode)
                print("✅ Code formatted with black")
            except ImportError:
                print("⚠️ black not installed. Install with: pip install black")
            except Exception as e:
                print(f"⚠️ Error formatting with black: {str(e)}")

        return formatted_code


class UnitTestGenerator:
    """
    Advanced AI-powered unit test generator using SambaNova with LangGraph.
    - Analyzes Python code to extract structure and determine test cases
    - Generates comprehensive test suites with mocking support
    - Validates and refines test cases
    - Supports user-provided test cases
    """

    def __init__(self, model_name="Qwen2.5-Coder-32B-Instruct", config=None):
        """Initialize the AI model and LangGraph workflow with configuration options."""
        load_dotenv()
        self.api_key = os.getenv("SAMBANOVA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SambaNova API key not found. Please set it in a .env file."
            )

        self.llm = ChatSambaNovaCloud(model=model_name, api_key=self.api_key)
        self.config = config or {
            "test_framework": "pytest",  # pytest, unittest
            "include_mocks": True,  # Generate mocks for dependencies
            "use_parameterized_tests": True,  # Use pytest parameterize
            "format_code": True,  # Format code with black
        }

        self.workflow = self.create_graph()

    def analyze_function(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """AI analyzes the function and returns test cases to cover with enhanced structure analysis."""
        code_to_test = state["code_to_test"]
        module_name = state.get("module_name")
        module_path = state.get("module_path")
        user_test_cases = state.get("user_test_cases", [])

        # Extract function structure information
        function_info = CodeParser.extract_function_info(code_to_test)
        if "error" in function_info:
            function_info = {"parsed": False}
        else:
            function_info["parsed"] = True

        # If user provided test cases, use them directly
        if (
            user_test_cases
            and isinstance(user_test_cases, dict)
            and "test_cases" in user_test_cases
        ):
            return {
                "analysis": user_test_cases,
                "code_to_test": code_to_test,
                "function_info": function_info,
                "module_name": module_name,
                "module_path": module_path,
            }

        # If user provided test suggestions but not a complete test cases dictionary
        user_suggestions = ""
        if user_test_cases:
            if isinstance(user_test_cases, list):
                user_suggestions = "\n".join(
                    [f"- {case}" for case in user_test_cases]
                )
            elif isinstance(user_test_cases, str):
                user_suggestions = user_test_cases

            user_suggestions = f"""
            User test case suggestions:
            {user_suggestions}
            
            Incorporate these suggestions into your analysis.
            """

        prompt = f"""
            You are an expert Python engineer specializing in test design. Analyze the given function and return a structured list of test cases.

            Function:
            ```python
            {code_to_test}
            ```
            
            Function Details:
            {json.dumps(function_info, indent=2)}
            
            {user_suggestions}

            Provide:
            1. A comprehensive list of test cases including:
            - Normal cases
            - Edge cases
            - Error cases
            - Performance considerations
            2. Expected outputs for each test case
            Important:

                • Do not include any Python operations or expressions (for example, do not return any concatenation or multiplication operations).
                • Instead, return only fully evaluated literal values. For instance, if a value is normally computed by ("a" * 1000), you must return the actual string of 1000 "a" characters.
                • Please return a simple JSON object containing only the test cases. Each test case should include:

                    name: A unique name for the test case
                    inputs: An object representing the input parameters for the function
                    expected_output: The expected output when the function is called with these inputs
                    description: A short description of the test case
                    Return only valid JSON with the following structure: {{ "test_cases": [ {{ "name": "test_example", "inputs": {{}}, "expected_output": null, "description": "A brief description of the test case" }} ] }} 
                • the json object should not contain any comments or additional text.
                • All values must be literal. For instance, if a value is derived from an expression like "a" * 1000, compute and return the final string instead of the expression. 
            """

        response = self.llm.invoke(prompt)
        try:
            json_str = re.search(
                r"```(?:json)?\s*({.*})\s*```", response.content, re.DOTALL
            )
            if json_str:
                json_str = json_str.group(1)
            else:
                json_str = response.content.strip()
        except Exception as e:
            json_str = response.content.strip()

        json_str = re.sub(r'(?<!")\bNone\b(?!")', "null", json_str)
        json_str = re.sub(r'(?<!")\bTrue\b(?!")', "true", json_str)
        json_str = re.sub(r'(?<!")\bFalse\b(?!")', "false", json_str)
        try:
            test_cases = demjson3.decode(json_str)
        except Exception as e:
            print(f"Error decoding JSON: {str(e)}")
            test_cases = json_str
        return {
            "analysis": test_cases,
            "code_to_test": code_to_test,
            "function_info": function_info,
            "module_name": module_name,
            "module_path": module_path,
        }

    def generate_tests(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """AI generates comprehensive unit tests based on function analysis."""
        code_to_test = state["code_to_test"]
        analysis = state["analysis"]
        function_info = state["function_info"]
        module_name = state.get("module_name")
        module_path = state.get("module_path")

        test_framework = self.config["test_framework"]
        use_parameterized = self.config["use_parameterized_tests"]
        include_mocks = self.config["include_mocks"]

        import_info = ""
        if module_name:
            import_info = f"Import the function from module: {module_name}"
        elif module_path:
            import_info = f"Import the function from file: {module_path}"

        prompt = f"""
            You are a senior Python engineer with expertise in testing. Based on the function and analysis below, generate {test_framework} unit tests.

            Function:
            ```python
            {code_to_test}
            ```
        
            Function structure:
            {json.dumps(function_info, indent=2)}

            Analysis of function:
            {json.dumps(analysis, indent=2)}

            Configuration:
            - Testing framework: {test_framework}
            - Use parameterized tests: {use_parameterized}
            - Include mocks: {include_mocks}
            {import_info}

            Important: Your generated tests should NOT include imports for the function being tested - those will be added automatically.
            Focus on writing the test functions only, assuming the function is already imported.

            Ensure:
            - Test cases cover all identified scenarios
            - Include proper assertions for expected results
            - Use descriptive test names that explain the purpose
            - Add proper docstrings to test functions
            - Use fixtures and setup/teardown appropriately
            - Handle potential exceptions correctly
            - If using parameterization, group similar tests
            - Include appropriate mocks for external dependencies
            - Follow best practices for {test_framework}

            Return only valid Python code inside triple backticks.
            """

        response = self.llm.invoke(prompt)

        code_blocks = self._extract_code_blocks(response.content)

        test_code = (
            "\n\n".join(code_blocks)
            if code_blocks
            else response.content.strip()
        )

        return {
            "generated_tests": test_code,
            "code_to_test": code_to_test,
            "analysis": analysis,
            "function_info": function_info,
            "module_name": module_name,
            "module_path": module_path,
        }

    def validate_tests(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """AI validates and refines test cases."""
        test_code = state["generated_tests"]
        code_to_test = state["code_to_test"]
        analysis = state["analysis"]
        function_info = state["function_info"]
        module_name = state.get("module_name")
        module_path = state.get("module_path")

        prompt = f"""
        You are a Python testing expert. Review the following test cases for:
        - Test coverage: Ensure all functionality is tested
        - Edge cases: Make sure all edge cases are covered
        - Code quality: Ensure tests follow best practices
        - Mocking: Check that external dependencies are properly mocked
        - Correctness: Ensure assertions are valid

        Original function:
        ```python
        {code_to_test}
        ```

        Test code:
        ```python
        {test_code}
        ```

        Analysis of function:
        {json.dumps(analysis, indent=2)}

        If there are issues:
        1. Provide a corrected version of the test code
        2. Explain what was missing or incorrect

        Return the corrected test code inside python code blocks.
        """

        response = self.llm.invoke(prompt)

        # Extract code blocks with improved regex pattern
        code_blocks = self._extract_code_blocks(response.content)

        # If no code blocks found, try to extract the code using a more flexible approach
        if not code_blocks:
            print(
                "⚠️ Warning: No code blocks found in validation response. Attempting extraction."
            )

            # First, look for markdown code blocks with different formatting
            code_blocks = re.findall(
                r"```(?:python)?\s*(.*?)```", response.content, re.DOTALL
            )

            # If still no code blocks found, try to extract Python-like code
            if not code_blocks:
                code_blocks = self._extract_python_like_code(response.content)

            # If all extraction methods failed, use the original test code
            if not code_blocks:
                print(
                    "⚠️ Warning: Failed to extract code from validation response. Using original test code."
                )
                code_blocks = [test_code]

        validated_tests = (
            "\n\n".join(code_blocks)
            if code_blocks
            else test_code  # Fall back to original test code
        )

        return {
            "validated_tests": validated_tests,
            "function_info": function_info,
            "module_name": module_name,
            "module_path": module_path,
        }

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks using multiple regex patterns to handle different formats."""
        # Try different regex patterns to extract code blocks
        patterns = [
            # Standard markdown code blocks with python language
            r"```python\s*(.*?)```",
            # Standard markdown code blocks without language specification
            r"```\s*(.*?)```",
            # Code blocks with different quote styles
            r"`{3}python\s*(.*?)`{3}",
            r"`{3}\s*(.*?)`{3}",
            # Python-like function definitions
            r"(?:^|\n)(def\s+test_.*?:.*?)(?=\n\S|$)",
        ]

        for pattern in patterns:
            code_blocks = re.findall(pattern, content, re.DOTALL)
            if code_blocks:
                # Clean up code blocks (remove leading/trailing whitespace)
                return [block.strip() for block in code_blocks]

        # If no code blocks found with standard patterns, try a more lenient approach
        return self._extract_python_like_code(content)

    def _extract_python_like_code(self, content: str) -> List[str]:
        """Extract Python-like code from the content."""
        # Look for sections that look like Python code with indentation patterns
        # This is a more aggressive approach when standard patterns fail

        # First try to find sections that start with "def test_" and continue until next def or end
        python_blocks = []

        # Try to extract what looks like test functions
        test_function_matches = re.findall(
            r'def\s+test_\w+\s*\(.*?\).*?:\s*(?:""".*?""")?.*?(?=\n\s*def|\Z)',
            content,
            re.DOTALL,
        )
        if test_function_matches:
            python_blocks.extend(test_function_matches)

        # Try to extract what looks like test fixtures
        fixture_matches = re.findall(
            r'@pytest\.fixture.*?\ndef\s+\w+\s*\(.*?\).*?:\s*(?:""".*?""")?.*?(?=\n\s*@|\n\s*def|\Z)',
            content,
            re.DOTALL,
        )
        if fixture_matches:
            python_blocks.extend(fixture_matches)

        # Try to find parameterize decorators
        parameterize_matches = re.findall(
            r'@pytest\.mark\.parametrize.*?\ndef\s+test_\w+\s*\(.*?\).*?:\s*(?:""".*?""")?.*?(?=\n\s*@|\n\s*def|\Z)',
            content,
            re.DOTALL,
        )
        if parameterize_matches:
            python_blocks.extend(parameterize_matches)

        # If still no blocks, try to extract any code that looks like Python (more aggressive)
        if not python_blocks:
            # Look for indented blocks that might be Python code
            lines = content.split("\n")
            in_code_block = False
            current_block = []

            for line in lines:
                if line.strip().startswith("def ") or line.strip().startswith(
                    "@"
                ):
                    # Start of a new potential Python code block
                    if current_block:
                        python_blocks.append("\n".join(current_block))
                        current_block = []
                    in_code_block = True
                    current_block.append(line)
                elif in_code_block:
                    if (
                        line.strip()
                        and not line.startswith("    ")
                        and not line.startswith("\t")
                        and not line.strip().startswith("#")
                    ):
                        # End of indented block
                        if current_block:
                            python_blocks.append("\n".join(current_block))
                            current_block = []
                        in_code_block = False
                    else:
                        current_block.append(line)

            # Add the last block if there is one
            if current_block:
                python_blocks.append("\n".join(current_block))

        return python_blocks

    def create_graph(self):
        """Create AI agent workflow using LangGraph."""
        workflow = StateGraph(dict)

        # Define AI-powered agents
        workflow.add_node("analyze", self.analyze_function)
        workflow.add_node("generate", self.generate_tests)
        workflow.add_node("validate", self.validate_tests)

        # Set initial entry point
        workflow.set_entry_point("analyze")

        # Connect nodes in sequence
        workflow.add_edge("analyze", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)

        return workflow.compile()

    def run(
        self,
        code_to_test: str,
        module_name: Optional[str] = None,
        module_path: Optional[str] = None,
        user_test_cases: Optional[
            Union[List[str], Dict[str, Any], str]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full pipeline with progress reporting.

        Args:
            code_to_test: Python code to generate tests for
            module_name: Optional module name for imports
            module_path: Optional module path for imports
            user_test_cases: Optional user-provided test cases or suggestions. Can be:
                - A complete dictionary with test_cases key
                - A list of test case descriptions
                - A string with test case suggestions
        """
        print("🔍 Analyzing function...")

        # Start the workflow
        initial_state = {
            "code_to_test": code_to_test,
            "module_name": module_name,
            "module_path": module_path,
            "user_test_cases": user_test_cases,
        }

        final_state = self.workflow.invoke(initial_state)

        function_name = "unknown_function"
        if "function_info" in final_state and final_state["function_info"].get(
            "parsed", False
        ):
            function_name = final_state["function_info"].get(
                "name", "unknown_function"
            )

        validated_tests = final_state["validated_tests"]
        if self.config.get("format_code", False):
            validated_tests = TestOutputFormatter.format_code(validated_tests)

        complete_test_file = TestOutputFormatter.format_test_file(
            function_name=function_name,
            test_code=validated_tests,
            module_name=final_state.get("module_name"),
            module_path=final_state.get("module_path"),
        )

        print(f"✅ Test generation complete for: {function_name}")

        return {
            "test_code": complete_test_file,
            "function_name": function_name,
            "function_info": final_state.get("function_info", {}),
        }

    def save_tests(
        self,
        test_code: str,
        output_file: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> str:
        """Save generated tests to a file."""
        if not output_file and function_name:
            output_file = f"test_{function_name}.py"
        elif not output_file:
            output_file = "generated_test.py"

        with open(output_file, "w") as f:
            f.write(test_code)

        print(f"✅ Tests saved to: {output_file}")
        return output_file


if __name__ == "__main__":
    # Sample functions for testing
    sample_function = """
def calculate_discount(price, discount_percentage):
    if not isinstance(price, (int, float)) or price < 0:
        raise ValueError("Price must be a positive number")
    
    if not isinstance(discount_percentage, (int, float)) or not 0 <= discount_percentage <= 100:
        raise ValueError("Discount percentage must be between 0 and 100")
    
    discount_amount = price * (discount_percentage / 100)
    return price - discount_amount
    """

    # Example of user-provided test cases using a list of suggestions
    user_test_suggestions = [
        "Test with regular price and 10% discount",
        "Test with zero price and 0% discount",
    ]

    # Example of user-provided test cases using a structured dictionary
    user_test_cases_dict = {
        "test_cases": [
            {
                "name": "test_regular_price_10_percent_discount",
                "inputs": {"price": 100, "discount_percentage": 10},
                "expected_output": 90,
                "description": "Test with regular price and 10% discount",
                "category": "normal",
            },
            {
                "name": "test_zero_price_no_discount",
                "inputs": {"price": 0, "discount_percentage": 0},
                "expected_output": 0,
                "description": "Test with zero price and 0% discount",
                "category": "edge",
            },
        ]
    }

    # Initialize the generator
    generator = UnitTestGenerator()

    # Example 1: Run with test suggestions
    print("Example 1: Using test suggestions")
    result1 = generator.run(
        sample_function, user_test_cases=user_test_suggestions
    )

    # Example 2: Run with structured test cases
    print("\nExample 2: Using structured test cases")
    result2 = generator.run(
        sample_function, user_test_cases=user_test_cases_dict
    )

    # Example 3: Run without user test cases (AI generates everything)
    print("\nExample 3: Using AI-generated test cases")
    result3 = generator.run(sample_function)

    # Save the generated tests
    generator.save_tests(
        result1["test_code"],
        function_name=f"{result1['function_name']}_with_suggestions",
    )
