import subprocess
import shlex
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field


mcp = FastMCP()

@mcp.tool(name="run_shell", description="Run a shell command")
def run_shell_command(command:
    Annotated[str, Field(description="shell command will be executed", examples=["ls -al"])]) -> str:
    try:
        shell_command = shlex.split(command)
        if "rm" in shell_command:
            raise Exception("不允许使用rm")
        if "del" in shell_command:
            raise Exception("不允许使用del")

        res = subprocess.run(command, shell=True, capture_output=True, text=True)

        if res.returncode != 0:
            return res.stderr
        return res.stdout
    except Exception as e:
        return str(e)


def run_shell_command_by_popen(commands):
    p = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    stdout, stderr = p.communicate()
    if stdout:
        return stdout
    return stderr


if __name__ == "__main__":
    mcp.run(transport="stdio")