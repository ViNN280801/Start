class InternalLogger:
    
    @staticmethod
    def pretty_function_details() -> str:
        """
        Prints the details of the calling function stack in the format:
        <file name>: line[<line number>]: <func name>(<args>)
        """
        from inspect import currentframe, getargvalues
        
        stack_details = []
        current_frame = currentframe()
        caller_frame = current_frame.f_back

        idx = 0
        while caller_frame is not None:
            function_name = caller_frame.f_code.co_name
            args, _, _, values = getargvalues(caller_frame)
            function_file = caller_frame.f_code.co_filename
            function_line_no = caller_frame.f_lineno
            formatted_args = ', '.join([f"{arg}={values[arg]}" for arg in args])
            stack_details.append(f"Call {idx}: {function_file}: line[{function_line_no}]: {function_name}({formatted_args})")
            caller_frame = caller_frame.f_back
            idx += 1

        return "\n".join(stack_details)

    @staticmethod
    def get_warning_none_result():
        return f"Warning, {InternalLogger.pretty_function_details()} returned '{None}' result"

    @staticmethod
    def get_warning_none_result_with_exception_msg(exmsg: str):
        return f"Warning, {InternalLogger.pretty_function_details()} returned '{None}' result. Exception: {exmsg}"
