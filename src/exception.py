"In this File Custom Exception Is Bulid so We Can use This as Package Anywhere Try Exception Is Raised"
import sys
import logging

def error_msg_detail(error,error_detail:sys):
    _,_,exc_tb =error_detail.exc_info()
    FileName = exc_tb.tb_frame.f_code.co_filename
    error_msg = "An Error Occured In Python Script Name [{0}] in the line [{1}] Error Message State's That [{2}]".format(
    FileName,exc_tb.tb_lineno,str(error))
    return error_msg

class CustomException(Exception):
    def __init__(self,error_msg,error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg,error_detail=error_detail)

    def __str__(self):
        return self.error_msg
