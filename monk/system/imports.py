import os
import sys
import shutil
import json
import pandas as pd
import numpy as np
import logging
import datetime
import functools
import inspect
import string
import warnings

from pylg import TraceFunction
from pylg import trace

  
class ArgumentValidationError(ValueError):
    def __init__(self, arg_num, func_name, accepted_arg_type, given_arg_type, list_type):
        if(list_type):
            self.error = 'The {0} argument of {1}() is not in the list {2}, but is {3}'.format(arg_num,
                                                                         func_name,
                                                                         accepted_arg_type,
                                                                         given_arg_type)
        else:
            self.error = 'The {0} argument of {1}() is not a {2}, but is {3}'.format(arg_num,
                                                                     func_name,
                                                                     accepted_arg_type,
                                                                     given_arg_type)
 
    def __str__(self):
        return self.error
 
 
 
class InvalidArgumentNumberError(ValueError):
    def __init__(self, func_name):
        self.error = 'Invalid number of arguments for {0}()'.format(func_name)
 
    def __str__(self):
        return self.error
 
 
 
class InvalidReturnType(ValueError):
    def __init__(self, return_type, func_name):
        self.error = 'Invalid return type {0} for {1}()'.format(return_type,
                                                                func_name)
 
    def __str__(self):
        return self.error



def ordinal(num):
    if 10 <= num % 100 < 20:
        return '{0}th'.format(num)
    else:
        ord = {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(num % 10, 'th')
        return '{0}{1}'.format(num, ord)


def accepts(*accepted_arg_types, **accepted_arg_dicts): 
    def accept_decorator(validate_function):
        @functools.wraps(validate_function)
        def decorator_wrapper(*function_args, **function_args_dicts):
            if len(accepted_arg_types) is not len(accepted_arg_types):
                raise InvalidArgumentNumberError(validate_function.__name__)
 
            # We're using enumerate to get the index, so we can pass the
            # argument number with the incorrect type to ArgumentValidationError.
            i = 0;
            for arg_num, (actual_arg, accepted_arg_type) in enumerate(zip(function_args, accepted_arg_types)):
                if(accepted_arg_type=="self"):
                    continue;
                if(type(accepted_arg_type)) == list:
                    if type(actual_arg) not in accepted_arg_type:
                        ord_num = ordinal(arg_num + 1)
                        if(accepted_arg_dicts["post_trace"]):
                            raise ArgumentValidationError(ord_num,
                                                      validate_function.function.function.__name__,
                                                      accepted_arg_type,
                                                      type(actual_arg),
                                                      True)
                        else:
                            raise ArgumentValidationError(ord_num,
                                                      validate_function.__name__,
                                                      accepted_arg_type,
                                                      type(actual_arg),
                                                      True)
                else:     
                    if not type(actual_arg) is accepted_arg_type:
                        ord_num = ordinal(arg_num + 1)
                        if(accepted_arg_dicts["post_trace"]):
                            raise ArgumentValidationError(ord_num,
                                                      validate_function.function.function.__name__,
                                                      accepted_arg_type,
                                                      type(actual_arg),
                                                      False)
                        else:
                            raise ArgumentValidationError(ord_num,
                                                      validate_function.__name__,
                                                      accepted_arg_type,
                                                      type(actual_arg),
                                                      False)
                i += 1;


            keys = list(function_args_dicts.keys());
            for i in range(len(keys)):
                func_arg_type = type(function_args_dicts[keys[i]]);
                accepted_arg_type = accepted_arg_dicts[keys[i]];
                if(type(accepted_arg_type) == list):
                    if(func_arg_type not in accepted_arg_type):
                        if(accepted_arg_dicts["post_trace"]):
                            raise ArgumentValidationError(keys[i],
                                                      validate_function.function.function.__name__,
                                                      accepted_arg_type,
                                                      func_arg_type,
                                                      True)
                        else:
                            raise ArgumentValidationError(keys[i],
                                                      validate_function.__name__,
                                                      accepted_arg_type,
                                                      func_arg_type,
                                                      True)
                else:
                    if(func_arg_type != accepted_arg_type):
                        if(accepted_arg_dicts["post_trace"]):
                            raise ArgumentValidationError(keys[i],
                                                      validate_function.function.function.__name__,
                                                      accepted_arg_type,
                                                      func_arg_type,
                                                      False)
                        else:
                            raise ArgumentValidationError(keys[i],
                                                      validate_function.__name__,
                                                      accepted_arg_type,
                                                      func_arg_type,
                                                      False)

            return validate_function(*function_args, **function_args_dicts)
        return decorator_wrapper
    return accept_decorator




class ConstraintError(ValueError):

    def __init__(self, msg):
        self.error = msg
 
    def __str__(self):
        return self.error



def ConstraintWarning(msg):
    warnings.warn(msg)




def error_checks(*arg_constraints, **kwargs_constraints):

    def check_gte(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value < limit):
                msg += "Value expected to be greater than equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] < limit):
                    msg += "List's arg number \"{}\" expected to be greater than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    raise ConstraintError(msg);



    def check_gt(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value <= limit):
                msg += "Value expected to be strictly greater than to \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] <= limit):
                    msg += "List's arg number \"{}\" expected to be strictly greater than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    raise ConstraintError(msg);



    def check_lte(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value > limit):
                msg += "Value expected to be less than equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] > limit):
                    msg += "List's arg number \"{}\" expected to be less than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    raise ConstraintError(msg);



    def check_lt(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value >= limit):
                msg += "Value expected to be strictly less than to \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] >= limit):
                    msg += "List's arg number \"{}\" expected to be strictly less than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    raise ConstraintError(msg);



    def check_eq(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float, str, list, tuple]):
            if(actual_value != limit):
                msg += "Value expected to be strictly equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);

    def check_neq(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float, str, list, tuple]):
            if(actual_value == limit):
                msg += "Value expected to be strictly not equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);

    def check_in(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in list(map(type, limit))):
            if(actual_value not in limit):
                msg += "Value expected to be one among \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);

    def check_nin(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in list(map(type, limit))):
            if(actual_value in limit):
                msg += "Value expected to be anything except \"{}\", but is \"{}\"".format(limit, actual_value);
                raise ConstraintError(msg);

    def check_folder(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == str):
            if(not os.path.isdir(actual_value)):
                msg = "Folder \"{}\" not found".format(actual_value)
                raise ConstraintError(msg);
            if(limit == "r"):
                if(not os.access(actual_value, os.R_OK)):
                    msg = "Folder \"{}\" has no read access".format(actual_value)
                    raise ConstraintError(msg);
            if(limit == "w"):
                if(not os.access(actual_value, os.W_OK)):
                    msg = "Folder \"{}\" has no write access".format(actual_value)
                    raise ConstraintError(msg);
        if(type(actual_value) == list):
            for i in range(len(actual_value)):
                if(not os.path.isdir(actual_value[i])):
                    msg = "Folder \"{}\" not found".format(actual_value[i])
                    raise ConstraintError(msg);
                if(limit == "r"):
                    if(not os.access(actual_value[i], os.R_OK)):
                        msg = "Folder \"{}\" has no read access".format(actual_value[i])
                        raise ConstraintError(msg);
                if(limit == "w"):
                    if(not os.access(actual_value[i], os.W_OK)):
                        msg = "Folder \"{}\" has no write access".format(actual_value[i])
                        raise ConstraintError(msg);


    def check_file(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == str):
            if(not os.path.isfile(actual_value)):
                msg = "File \"{}\" not found".format(actual_value)
                raise ConstraintError(msg);
        
            if(limit == "r"):
                if(not os.access(actual_value, os.R_OK)):
                    msg = "File \"{}\" has no read access".format(actual_value)
                    raise ConstraintError(msg);
            if(limit == "w"):
                if(not os.access(actual_value, os.W_OK)):
                    msg = "File \"{}\" has no write access".format(actual_value)
                    raise ConstraintError(msg);
        if(type(actual_value) == list):
            for i in range(len(actual_value)):
                if(not os.path.isfile(actual_value[i])):
                    msg = "File \"{}\" not found".format(actual_value[i])
                    raise ConstraintError(msg);
            
                if(limit == "r"):
                    if(not os.access(actual_value[i], os.R_OK)):
                        msg = "File \"{}\" has no read access".format(actual_value[i])
                        raise ConstraintError(msg);
                if(limit == "w"):
                    if(not os.access(actual_value[i], os.W_OK)):
                        msg = "File \"{}\" has no write access".format(actual_value[i])
                        raise ConstraintError(msg);


    def check_inc(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == list):
            if(sorted(actual_value) != actual_value):
                msg += "List expected to be incremental, but is \"{}\"".format(actual_value);
                raise ConstraintError(msg);

    def check_dec(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == list):
            if(sorted(actual_value, reverse=True) != actual_value):
                msg += "List expected to be decremental, but is \"{}\"".format(actual_value);
                raise ConstraintError(msg);


    def check_name(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == str):
            total_list = [];
            for i in range(len(limit)):
                if(limit[i] == "a-z"):
                    total_list += list(string.ascii_lowercase)
                elif(limit[i] == "A-Z"):
                    total_list += list(string.ascii_uppercase)
                elif(limit[i] == "0-9"):
                    total_list += list(string.digits)
                else:
                    total_list += limit[i];
            
            actual_value = list(actual_value)
            for j in range(len(actual_value)):
                if(actual_value[j] not in total_list):
                    msg += "Character \"{}\" not allowed as per constrains \"{}\"".format(actual_value[j], limit);
                    raise ConstraintError(msg);



    def accept_decorator(validate_function):
        @functools.wraps(validate_function)
        def decorator_wrapper(*function_args, **function_args_dicts):
            if len(arg_constraints) is not len(function_args):
                raise InvalidArgumentNumberError(validate_function.__name__)

            if(kwargs_constraints["post_trace"]):
                function_name = validate_function.function.function.__name__;
            else:
                function_name = validate_function.__name__;

            for arg_num, (actual_arg, arg_constraint) in enumerate(zip(function_args, arg_constraints)):
                if(arg_constraint):
                    for i in range(len(arg_constraint)//2):
                        if(arg_constraint[i*2] == "gte"):
                            check_gte(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "gt"):
                            check_gt(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "lte"):
                            check_lte(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "lt"):
                            check_lt(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "eq"):
                            check_eq(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "neq"):
                            check_neq(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "in"):
                            check_in(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "nin"):
                            check_nin(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "folder"):
                            check_folder(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "file"):
                            check_file(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "inc"):
                            check_inc(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "dec"):
                            check_dec(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "name"):
                            check_name(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);

                                 


            keys = list(function_args_dicts.keys());
            for x in range(len(keys)):
                actual_arg = function_args_dicts[keys[x]];
                arg_constraint = kwargs_constraints[keys[x]];
                if(arg_constraint):
                    for i in range(len(arg_constraint)//2):
                        if(arg_constraint[i*2] == "gte"):
                            check_gte(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "gt"):
                            check_gt(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "lte"):
                            check_lte(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "lt"):
                            check_lt(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "eq"):
                            check_eq(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "neq"):
                            check_neq(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "in"):
                            check_in(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "nin"):
                            check_nin(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "folder"):
                            check_folder(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "file"):
                            check_file(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "inc"):
                            check_inc(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "dec"):
                            check_dec(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "name"):
                            check_name(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);



            return validate_function(*function_args, **function_args_dicts)
        return decorator_wrapper
    return accept_decorator



def warning_checks(*arg_constraints, **kwargs_constraints):

    def check_gte(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value < limit):
                msg += "Value expected to be greater than equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] < limit):
                    msg += "List's arg number \"{}\" expected to be greater than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    ConstraintWarning(msg);


    def check_gt(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value <= limit):
                msg += "Value expected to be strictly greater than to \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] <= limit):
                    msg += "List's arg number \"{}\" expected to be strictly greater than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    ConstraintWarning(msg);



    def check_lte(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value > limit):
                msg += "Value expected to be less than equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] > limit):
                    msg += "List's arg number \"{}\" expected to be less than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    ConstraintWarning(msg);



    def check_lt(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float]):
            if(actual_value >= limit):
                msg += "Value expected to be strictly less than to \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);
        if(type(actual_value) in [list, tuple]):
            for i in range(len(actual_value)):
                if(actual_value[i] >= limit):
                    msg += "List's arg number \"{}\" expected to be strictly less than equal to \"{}\", but is \"{}\"".format(i+1, limit, actual_value[i]);
                    ConstraintWarning(msg);


    def check_eq(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float, str, list, tuple]):
            if(actual_value != limit):
                msg += "Value expected to be strictly equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);

    def check_neq(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in [int, float, str, list, tuple]):
            if(actual_value == limit):
                msg += "Value expected to be strictly not equal to \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);

    def check_in(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in list(map(type, limit))):
            if(actual_value not in limit):
                msg += "Value expected to be one among \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);

    def check_nin(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) in list(map(type, limit))):
            if(actual_value in limit):
                msg += "Value expected to be anything except \"{}\", but is \"{}\"".format(limit, actual_value);
                ConstraintWarning(msg);

    def check_folder(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == str):
            if(not os.path.isdir(actual_value)):
                msg = "Folder \"{}\" not found".format(actual_value)
                ConstraintWarning(msg);
            if(limit == "r"):
                if(not os.access(actual_value, os.R_OK)):
                    msg = "Folder \"{}\" has no read access".format(actual_value)
                    ConstraintWarning(msg);
            if(limit == "w"):
                if(not os.access(actual_value, os.W_OK)):
                    msg = "Folder \"{}\" has no write access".format(actual_value)
                    ConstraintWarning(msg);
        if(type(actual_value) == list):
            for i in range(len(actual_value)):
                if(not os.path.isdir(actual_value[i])):
                    msg = "Folder \"{}\" not found".format(actual_value[i])
                    ConstraintWarning(msg);
                if(limit == "r"):
                    if(not os.access(actual_value[i], os.R_OK)):
                        msg = "Folder \"{}\" has no read access".format(actual_value[i])
                        ConstraintWarning(msg);
                if(limit == "w"):
                    if(not os.access(actual_value[i], os.W_OK)):
                        msg = "Folder \"{}\" has no write access".format(actual_value[i])
                        ConstraintWarning(msg);


    def check_file(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == str):
            if(not os.path.isdir(actual_value)):
                msg = "File \"{}\" not found".format(actual_value)
                ConstraintWarning(msg);
            if(limit == "r"):
                if(not os.access(actual_value, os.R_OK)):
                    msg = "File \"{}\" has no read access".format(actual_value)
                    ConstraintWarning(msg);
            if(limit == "w"):
                if(not os.access(actual_value, os.W_OK)):
                    msg = "File \"{}\" has no write access".format(actual_value)
                    ConstraintWarning(msg);
        if(type(actual_value) == list):
            for i in range(len(actual_value)):
                if(not os.path.isdir(actual_value[i])):
                    msg = "File \"{}\" not found".format(actual_value[i])
                    ConstraintWarning(msg);
                if(limit == "r"):
                    if(not os.access(actual_value[i], os.R_OK)):
                        msg = "File \"{}\" has no read access".format(actual_value[i])
                        ConstraintWarning(msg);
                if(limit == "w"):
                    if(not os.access(actual_value[i], os.W_OK)):
                        msg = "File \"{}\" has no write access".format(actual_value[i])
                        ConstraintWarning(msg);


    def check_inc(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == list):
            if(sorted(actual_value) != actual_value):
                msg += "List expected to be incremental, but is \"{}\"".format(actual_value);
                ConstraintWarning(msg);

    def check_dec(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == list):
            if(sorted(actual_value, reverse=True) != actual_value):
                msg += "List expected to be decremental, but is \"{}\"".format(actual_value);
                ConstraintWarning(msg);


    def check_name(actual_value, limit, function_name, arg_num=None, arg_name=None):
        if(arg_num):
            arg = arg_num;
            msg = "Constraint Mismatch for argument number \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(arg_name):
            arg = arg_name;
            msg = "Constraint Mismatch for argument name \"{}\" in function \"{}\".\n".format(arg, function_name);
        if(type(actual_value) == str):
            total_list = [];
            for i in range(len(limit)):
                if(limit[i] == "a-z"):
                    total_list += list(string.ascii_lowercase)
                elif(limit[i] == "A-Z"):
                    total_list += list(string.ascii_uppercase)
                elif(limit[i] == "0-9"):
                    total_list += list(string.digits)
                else:
                    total_list += limit[i];
            
            actual_value = list(actual_value)
            for j in range(len(actual_value)):
                if(actual_value[j] not in total_list):
                    msg += "Character \"{}\" not allowed as per constrains \"{}\"".format(actual_value[j], limit);
                    ConstraintWarning(msg);



    def accept_decorator(validate_function):
        @functools.wraps(validate_function)
        def decorator_wrapper(*function_args, **function_args_dicts):
            if len(arg_constraints) is not len(function_args):
                raise InvalidArgumentNumberError(validate_function.__name__)

            if(kwargs_constraints["post_trace"]):
                function_name = validate_function.function.function.__name__;
            else:
                function_name = validate_function.__name__;

            for arg_num, (actual_arg, arg_constraint) in enumerate(zip(function_args, arg_constraints)):
                if(arg_constraint):
                    for i in range(len(arg_constraint)//2):
                        if(arg_constraint[i*2] == "gte"):
                            check_gte(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "gt"):
                            check_gt(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "lte"):
                            check_lte(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "lt"):
                            check_lt(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "eq"):
                            check_eq(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "neq"):
                            check_neq(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "in"):
                            check_in(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "nin"):
                            check_nin(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "folder"):
                            check_folder(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "file"):
                            check_file(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "inc"):
                            check_inc(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "dec"):
                            check_dec(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);
                        if(arg_constraint[i*2] == "name"):
                            check_name(actual_arg, arg_constraint[i*2+1], function_name, arg_num=arg_num+1);

                                 


            keys = list(function_args_dicts.keys());
            for x in range(len(keys)):
                actual_arg = function_args_dicts[keys[x]];
                arg_constraint = kwargs_constraints[keys[x]];
                if(arg_constraint):
                    for i in range(len(arg_constraint)//2):
                        if(arg_constraint[i*2] == "gte"):
                            check_gte(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "gt"):
                            check_gt(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "lte"):
                            check_lte(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "lt"):
                            check_lt(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "eq"):
                            check_eq(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "neq"):
                            check_neq(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "in"):
                            check_in(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "nin"):
                            check_nin(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "folder"):
                            check_folder(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "file"):
                            check_file(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "inc"):
                            check_inc(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "dec"):
                            check_dec(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);
                        if(arg_constraint[i*2] == "name"):
                            check_name(actual_arg, arg_constraint[i*2+1], function_name, arg_name=keys[x]);



            return validate_function(*function_args, **function_args_dicts)
        return decorator_wrapper
    return accept_decorator

