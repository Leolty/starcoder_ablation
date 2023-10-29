from typing import Union
import re


# crop lines given a threshold and tokenized code
def crop_code_lines(code: Union[str, list], threshold: int):
    """
    Crop the code to the last threshold lines.

    Args:
        code: the code to crop (either a string or a list of tokens)
        threshold: the number of lines to keep
    
    Returns:
        code: the cropped code
    """

    # if the code is string, meaning it is not tokenized
    if isinstance(code, str):
        code = code.split('\n')

        # return the last threshold lines if the code is longer than the threshold
        if len(code) > threshold:
            return "\n".join(code[-threshold:])
        else:
            return "\n".join(code)
    
    # if the code is tokenized
    elif isinstance(code, list):
        # set the current number of lines to -1, since the last line is always a newline
        cur_lines = -1
        # iterate over the code list from the end to the beginning
        for i in range(len(code)-1, -1, -1):
            # "Ċ" is the token for newline 
            if "Ċ" in code[i]:
                cur_lines += 1
            
            # if the current number of lines reaches the threshold, 
            # return the code from the current index to the end (do not include the newline token)
            if cur_lines == threshold:
                return code[i+1:]
        
        # if the code is shorter than the threshold, return the whole code
        return code

# comment code
def comment(code: str, language: str):
    """
    Comment the code.

    Args:
        code: the code to comment
        language: the language of the code
    
    Returns:
        code: the commented code
    """
    if language == "python":
        return "\n".join([f"# {line}" for line in code.split("\n")])
    elif language == "java":
        return "\n".join([f"// {line}" for line in code.split("\n")])
    else:
        raise ValueError("language must be one of [python, java]")

def normalize_empty_lines(code: str) -> str:
    """
    Normalize consecutive empty lines in a string to a maximum of two.

    Args:
        code (str): Code to normalize.
    
    Returns:
        str: Normalized code.
    """
    normalized_code = re.sub(r'\n{4,}', '\n\n', code)
    return normalized_code

def construct_prompt(data: dict, version: str = 'repo') -> Union[str, dict]:
    """
    Constructs a prompt for the specified model version.

    Args:
        data: the data to construct the prompt from
        version: 'repo', 'file', 'baseline', or 'all'
    
    Returns:
        prompt: the constructed prompt or a list of prompts if version is 'all'
    """
    prompts = {}

    # Repo version
    repo_name = data['repo_name']
    file_path = data['file_path']
    code = data['all_code']
    import_statement = data['import_statement']

    prompt_repo = f"<reponame>{repo_name}"
    for snippet in data['context']:
        prompt_repo += f"<filepath>{snippet['path']}\n{snippet['snippet']}"
    prompt_repo += f"<filepath>{file_path}\n{import_statement}\n{code}"
    prompts['repo'] = normalize_empty_lines(prompt_repo)

    # File version
    prompt_file = f"# Repo Name: {data['repo_name']}\n"
    for snippet in data['context']:
        prompt_file += f"# Path: {snippet['path']}\n{snippet['snippet']}" + "\n"
    prompt_file += f"# Path: {file_path}\n{import_statement}\n{code}"
    prompts['file'] = normalize_empty_lines(prompt_file)

    # Baseline version
    prompt_baseline = f"# Path: {file_path}\n{import_statement}\n{code}"
    prompts['baseline'] = normalize_empty_lines(prompt_baseline)

    if version == "all":
        return prompts
    else:
        return prompts[version]


def get_first_line_not_comment(code:str, language:str="python"):
    """
    This function gets the first line of code that is not a comment.

    Args:
    code: Str, the code

    Returns:
    Str, the first line of code that is not a comment or the first line of code if there is no line that is not a comment
    """

    # check if the language is valid
    assert language in ["python", "java"], "language must be one of [python, java]"


    # first remove the \n at the beginning of the code
    code = code.lstrip('\n')

    lines = code.split('\n')
    in_multiline_comment = False

    if language == "python":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('#'):
                continue
            # if the line is not a comment, then return the line
            return line
        
    elif language == "java":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and line.strip().startswith('/*'):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and line.strip().endswith('*/'):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('//'):
                continue
            # if the line is not a comment, then return the line
            return line


    # if we cannot find a line that is not a comment, then return the first line
    return lines[0]