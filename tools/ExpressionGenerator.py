import itertools
from collections import defaultdict



class ExpressionGenerator:
    """
    ExpressionGenerator is a class designed to generate mathematical expressions based on their complexity level.
    It supports various types of operators including unary, binary, window-based, and conditional operators.
    The generator creates expressions by recursively combining simpler expressions according to predefined rules.
    """

    def __init__(self, expression_save_path: str):
        """
        Initialize the ExpressionGenerator with a save path and operator configurations.
        """
        self.save_path = expression_save_path

        operator_complexity = {
            'abs': 1,
            'inv': 1,
            'neg': 1,
            'ln': 1,
            'sign': 1,
            'square': 1,
            'sign_square': 1,
            'power_e': 1,
            'sqrt': 1,
            'sign_sqrt': 1,
            'demean': 1,
            'demedian': 1,
            'max_val': 1,
            'min_val': 1,
            'cs_rank': 1,
            'cs_mean': 1,
            'cs_median': 1,
            'cs_sum': 1,
            'cs_std': 1,
            'shift_one': 1,
            'pct_change_one': 1,
            'diff_one': 1,
            'add': 1,
            'sub': 1,
            'mul': 1,
            'div': 1,
            'cs_corr': 1,
            'cs_cov': 1,
            'cs_ols_beta': 1,
            'cs_ols_resid': 1,
            'cs_ols_intercept': 1,
            'shift': 1,
            'pct_change': 1,
            'diff': 1,
            'ts_mean': 1,
            'ts_sum': 1,
            'ts_prod': 1,
            'ts_std': 1,
            'ts_max': 1,
            'ts_min': 1,
            'ts_argmax': 1,
            'ts_argmin': 1,
            'ts_rank': 2,
            'ts_trend_beta': 1,
            'ts_trend_intercept': 1,
            'ts_trend_resid_std': 1,
            'ts_corr': 1,
            'ts_cov': 1,
            'ts_ols_beta': 1,
            'ts_ols_intercept': 1,
            'ts_ols_resid_std': 1,
            'where': 2
        }

        self.operator_arity = {
            'abs': 1,
            'inv': 1,
            'neg': 1,
            'ln': 1,
            'sign': 1,
            'square': 1,
            'sign_square': 1,
            'power_e': 1,
            'sqrt': 1,
            'sign_sqrt': 1,
            'demean': 1,
            'demedian': 1,
            'cs_rank': 1,
            'cs_mean': 1,
            'cs_median': 1,
            'cs_sum': 1,
            'cs_std': 1,
            'shift_one': 1,
            'pct_change_one': 1,
            'diff_one': 1,
            'add': 2,
            'sub': 2,
            'mul': 2,
            'div': 2,
            'max_val': 2,
            'min_val': 2,
            'cs_corr': 2,
            'cs_cov': 2,
            'cs_ols_beta': 2,
            'cs_ols_resid': 2,
            'cs_ols_intercept': 2,
            'shift': 1,
            'pct_change': 1,
            'diff': 1,
            'ts_mean': 1,
            'ts_sum': 1,
            'ts_prod': 1,
            'ts_std': 1,
            'ts_max': 1,
            'ts_min': 1,
            'ts_argmax': 1,
            'ts_argmin': 1,
            'ts_rank': 1,
            'ts_trend_beta': 1,
            'ts_trend_intercept': 1,
            'ts_trend_resid_std': 1,
            'ts_corr': 2,
            'ts_cov': 2,
            'ts_ols_beta': 2,
            'ts_ols_intercept': 2,
            'ts_ols_resid_std': 2,
            'where': 4
        }

        self.simple_operators = [
            'abs', 'inv', 'neg', 'ln', 'sign', 'square', 'sign_square', 'power_e', 'sqrt', 
            'sign_sqrt', 'demean', 'demedian', 'cs_rank', 'cs_mean', 
            'cs_median', 'cs_sum', 'cs_std', 'shift_one', 'pct_change_one', 'diff_one', 
        ]

        self.binary_operators = [
            'add', 'sub', 'mul', 'div', 'max_val', 'min_val', 'cs_corr', 'cs_cov', 'cs_ols_beta', 
            'cs_ols_resid', 'cs_ols_intercept'
        ]
        
        self.window_operators = [
            'shift', 'pct_change', 'diff', 'ts_mean', 'ts_sum', 'ts_prod', 
            'ts_std', 'ts_max', 'ts_min', 'ts_argmax', 'ts_argmin', 
            'ts_rank', 'ts_trend_beta', 'ts_trend_intercept', 'ts_trend_resid_std', 
            'ts_corr', 'ts_cov', 'ts_ols_beta', 'ts_ols_intercept', 'ts_ols_resid_std'
        ]

        self.cond_operators = ['where']

        self.logic_operators = ['is_larger', 'is_smaller']

        self.base_variables = ['ret_df', 'turn_df', 'amount_df', 'volume_df', 'marketcap_df', 'excess_df', 'market_df', 
                               'open_df', 'high_df', 'low_df', 'close_df']
        
        self.fixed_windows = [5]

        self.operators_by_complexity = defaultdict(list)
        for op, comp in operator_complexity.items():
            self.operators_by_complexity[comp].append(op)

        self.operators_by_arity = defaultdict(list)
        for op, comp in self.operator_arity.items():
            self.operators_by_arity[comp].append(op)


    def generate_expressions(self, target_complexity: int):
        """
        Generate expressions with the specified target complexity.
        This method recursively generates expressions by combining simpler expressions
        using operators of appropriate complexity.
        """
        if target_complexity == 0:
            yield from self.base_variables
            return
        available_complexities = [comp for comp in self.operators_by_complexity.keys() 
                                if comp <= target_complexity]
        for chosen_complexity in available_complexities:
            operators = self.operators_by_complexity[chosen_complexity]
            for chosen_operator in operators:
                arity = self.operator_arity[chosen_operator]
                remain_complexity = target_complexity - chosen_complexity
                arg_complexity_pairs = self._get_arg_complexity_pairs(arity, remain_complexity)
                for pair in arg_complexity_pairs:
                    arg_generators = [self.generate_expressions(t) for t in pair]
                    for comb in itertools.product(*arg_generators):
                        exp = self._build_expression(chosen_operator, comb)
                        yield from exp


    def _get_arg_complexity_pairs(self, n: int, k: int):
        """
        Generate all possible ways to distribute complexity k among n arguments.
        This method finds all combinations of non-negative integers that sum to k,
        with exactly n numbers in each combination.
        """
        return [i for i in itertools.product(range(k+1), repeat=n) if sum(i)==k]


    def _build_expression(self, operator: str, args):
        """
        Construct an expression string based on the operator type and its arguments.
        Different operators require different formatting patterns.
        """
        if operator in self.simple_operators:
            return [f"{operator}({args[0]})"]
        
        elif operator in self.binary_operators:
            if operator in ['add', 'sub', 'mul', 'div']:
                operator = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[operator]
                return [f"({args[0]}{operator}{args[1]})"]
            else:
                return [f"{operator}({args[0]},{args[1]})"]
            
        elif operator in self.window_operators:
            if operator in self.operators_by_arity[1]:
                return [f"{operator}({args[0]},{window})" for window in self.fixed_windows]
            elif operator in self.operators_by_arity[2]:
                return [f"{operator}({args[0]},{args[1]},{window})" for window in self.fixed_windows]
            
        elif operator in self.cond_operators:
            if operator == 'cond':
                return [f"{operator}({logic_operator}({args[0]},{args[1]}),{args[2]})" 
                        for logic_operator in self.logic_operators]
            elif operator == 'where':
                return [f"{operator}({logic_operator}({args[0]},{args[1]}),{args[2]},{args[3]})" 
                        for logic_operator in self.logic_operators]
            
        else:
            raise ValueError(f"operator:{operator},args:{args}")