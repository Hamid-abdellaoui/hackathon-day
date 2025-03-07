import os
import re
import json
import ast
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_sambanova import ChatSambaNovaCloud
from langgraph.graph import StateGraph, END


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
    def extract_module_name(file_path: str) -> str:
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
        module_name: str = None,
        module_path: str = None,
        imports: List[str] = None,
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
        test_code: str, output_path: str = None, function_name: str = None
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

        # Extract function structure information
        function_info = CodeParser.extract_function_info(code_to_test)
        if "error" in function_info:
            function_info = {"parsed": False}
        else:
            function_info["parsed"] = True

        prompt = f"""
            You are an expert Python engineer specializing in test design. Analyze the given function and return a structured list of test cases.

            Function:
            ```python
            {code_to_test}
            ```
            
            Function Details:
            {json.dumps(function_info, indent=2)}

            Provide:
            1. A comprehensive list of test cases including:
            - Normal cases
            - Edge cases
            - Error cases
            - Performance considerations
            2. Expected outputs for each test case
            3. Required mocks or fixtures
            4. Potential parameterizations for similar tests
            
            Return only valid JSON format with the following structure:
            {{
                "test_cases": [
                    {{
                        "name": "test_name",
                        "inputs": {{...}},
                        "expected_output": value,
                        "description": "description",
                        "category": "normal|edge|error"
                    }}
                ],
                "mocks": [
                    {{
                        "target": "module.function",
                        "return_value": value
                    }}
                ],
                "parameterized_groups": [
                    {{
                        "name": "group_name",
                        "cases": [{{...}}]
                    }}
                ]
            }}
            """

        response = self.llm.invoke(prompt)

        json_match = re.search(
            r'```(?:json)?\s*({\s*"test_cases".*})\s*```',
            response.content,
            re.DOTALL,
        )

        if json_match:
            json_data = json_match.group(1)
            try:
                # Add error handling and cleaning before parsing
                json_data = json_data.strip()

                # Check if JSON is complete
                if json_data.count("{") != json_data.count("}"):
                    print("⚠️ Warning: Incomplete JSON detected")
                    # Complete the JSON structure
                    json_data = json_data.rsplit("},", 1)[0] + "}]}}}"

                # Remove any trailing commas before closing brackets
                json_data = re.sub(r",\s*}", "}", json_data)
                json_data = re.sub(r",\s*]", "]", json_data)

                test_cases = json.loads(json_data)

            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing JSON: {str(e)}")
                print("Falling back to default structure")
                test_cases = {"test_cases": []}
        else:
            print(
                "⚠️ Warning: No JSON structure found. Falling back to default structure."
            )
            test_cases = {"test_cases": []}

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
        code_blocks = re.findall(
            r"```python\n(.*?)```", response.content, re.DOTALL
        )
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
        code_blocks = re.findall(
            r"```python\n(.*?)```", response.content, re.DOTALL
        )
        validated_tests = (
            "\n\n".join(code_blocks)
            if code_blocks
            else response.content.strip()
        )

        return {
            "validated_tests": validated_tests,
            "function_info": function_info,
            "module_name": module_name,
            "module_path": module_path,
        }

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
        module_name: str = None,
        module_path: str = None,
    ) -> Dict[str, Any]:
        """Runs the full pipeline with progress reporting."""
        print("🔍 Analyzing function...")

        # Start the workflow
        initial_state = {
            "code_to_test": code_to_test,
            "module_name": module_name,
            "module_path": module_path,
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
        output_file: str = None,
        function_name: str = None,
    ):
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
    sample_function = """
def extract_ids_from_url(url, param_names: list[str]) -> dict:
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    return {param: query_params.get(param, [None])[0] for param in param_names}
    """
    sample_function = """
def get_value_formatter(col):
    if "net_" in col.lower() or col.lower() == "ca":
        return {"function": "params.value.toLocaleString() + ' €'"}
    elif "(€)" in col.lower():
        return {"function": "params.value.toLocaleString() + ' €'"}
    elif "share" in col.lower():
        return {"function": "params.value + ' %'"}
    else:
        return {"function": "params.value.toLocaleString()"}
"""

    generator = UnitTestGenerator()
    result = generator.run(sample_function)

    generator.save_tests(
        result["test_code"], function_name=result["function_name"]
    )
