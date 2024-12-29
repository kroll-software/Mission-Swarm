import os
import io
import contextlib
import ast
#import pkg_resources

def execute_python_code(code: str) -> str:
    print("Using Tool: execute_python_code")
    
    # convert line-breaks    
    code = code.replace("\\n", "\n")    

    try:
        ''' 
        node = ast.parse(code)        
        nodes = [n for n in node.body if isinstance(n, ast.FunctionDef) or isinstance(n, ast.ClassDef)]        
        safe_locals = {n.name: n for n in nodes}
        '''
        

        b = compile(code, 'something', 'exec')
        
        safe_globals = globals().copy()
        safe_globals.pop("os", None)
        safe_globals.pop("io", None)
        safe_globals.pop("pkg_resources", None)
        safe_globals.pop("contextlib", None)
        #print(safe_globals)
        
        #safe_locals = {}
        safe_locals = safe_globals        
                        
        output_stream = io.StringIO()
        with contextlib.redirect_stdout(output_stream):
            exec(b, safe_globals, safe_locals)

        result_string = output_stream.getvalue()
        if len(result_string) > 0:
            return result_string
        else:
            # Return the locals as a string for any assigned variables or expressions
            if len(safe_locals) < 2:            
                return str(safe_locals)
            else:
                last_key = list(safe_locals) [-1]        
                return f"'{last_key}' = {str(safe_locals[last_key])}"            
        
    except Exception as e:        
        msg = e.msg if hasattr(e, "msg") else str(e)
        return f"{msg}\n\nYour code was:\n\n{code}\n\n**Please revise your code**"
    

def list_installed_python_libraries() -> str:    
    ''' List all available python libs '''
    import pkg_resources
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    return "\n".join(installed_packages_list)