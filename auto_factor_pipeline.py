from tools.ExpressionGenerator import ExpressionGenerator
from tools.FactorTest import ExpressionTest
from tools.FactorBaseManager import FactorBaseManager
from Config import RootConfig

from tqdm import tqdm
import os
import glob
import csv
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")



"""
Automated Factor Generation Main Pipeline.

This pipeline runs the following iterative process:
1. Generate expressions
2. Round 1: Calculate and test expressions using 3-year small sample
3. Round 2: Calculate and test expressions using 2017-2024 sample
4. Store factors that pass both rounds of testing
5. When factor library exceeds 1000 factors, perform factor library management to control absolute correlation between factors below 70%
"""



class AutoFactorPipeLine:
    """
    AutoFactorPipeLine orchestrates the automated factor generation process.
    It manages expression generation, testing, and factor library maintenance
    to ensure quality and diversity of generated factors.
    """

    def __init__(self, factor_data_root: str):
        """
        Initialize the automated factor pipeline.
        """
        # Path to store the last tested expression, used for resuming from breakpoints
        self.last_exp_path = os.path.join(factor_data_root, "last_tested_exp.csv")
        # Path to store qualified expressions
        self.factor_save_path = os.path.join(factor_data_root, "factors")
        
        # Initialize expression generator
        self.generator = ExpressionGenerator(self.factor_save_path)
        # Initialize factor calculation and testing tool
        self.exp_test = ExpressionTest(factor_data_base=self.factor_save_path)


    def exps_generate(self, max_complexity: int, batch_size: int):
        """
        Find the last checkpoint and continue generating factor expressions.
        This method handles resuming from a previous run by finding the last tested expression.
        """
        last_exp, init_complexity = self._read_last_exp()
        batch = []

        if last_exp:
            found_last = False
        else:
            found_last = True

        # Generate expressions by complexity level
        for i in range(init_complexity, max_complexity + 1):
            for exp in self.generator.generate_expressions(i):
                if not found_last:
                    # Skip expressions until we find the last one from previous run
                    if exp == last_exp:
                        found_last = True
                    continue

                batch.append((exp, i))

                # Return batch when it reaches the specified size
                if len(batch) >= batch_size:
                    return batch, False
                
            print(f"\nComplexity {i} finished!")
        
        # Return remaining expressions if any
        return batch, True


    def run_test_exps(self, batch, max_workers: int):
        """
        Multi-process calculation and testing of expressions.
        This method processes a batch of expressions in parallel.
        """
        exps = []
        for i in batch:
            exps.append(i[0])
        
        # Process expressions in parallel
        with Pool(processes=max_workers) as pool:
            for _ in pool.imap_unordered(self._process_expression, exps):
                pass
            # Record the last processed expression
            self._write_last_exp(batch[-1])


    def factor_base_maintain(self, max_files: int):
        """
        Maintain the factor base by removing highly correlated factors.
        This helps keep the factor library diverse and manageable.
        """
        factor_base_manager = FactorBaseManager(self.factor_save_path, max_files)
        factor_base_manager.remove_high_coor_factors()


    def _read_last_exp(self):
        """
        Read the last tested expression from file.
        """
        try: 
            with open(self.last_exp_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    return row[0], int(row[1])
        except:
            # If no previous record exists, start from complexity level 1
            return None, 1


    def _write_last_exp(self, data):
        """
        Write the last tested expression to file (overwrite).
        This enables resuming from the correct position if interrupted.
        """
        with open(self.last_exp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)


    def _process_expression(self, exp):
        """
        Run calculation and testing for a single expression.
        This method handles the actual testing of an individual expression.
        """
        try:
            self.exp_test.run(exp)
        except Exception as e:
            print(f"Error processing {exp}: {e}")


if __name__ == '__main__':
    # Initialize the pipeline
    pipe_line = AutoFactorPipeLine(RootConfig.FACTOR_DATA_ROOT)
    
    # Configuration parameters
    max_complexity = 10          # Maximum complexity level for generated expressions
    batch_size = 1000            # Number of expressions to process in each batch
    max_files = 500              # Threshold for factor library maintenance
    finished = False             # Flag to indicate completion status

    # Progress bar for expression generation
    with tqdm(desc="Expression Generation Progress", unit="exp") as pbar:
        while not finished:
            # Generate a batch of expressions
            batch, finished = pipe_line.exps_generate(max_complexity, batch_size)
            
            # Test the batch of expressions in parallel
            pipe_line.run_test_exps(batch, max_workers=8)
            pbar.update(batch_size)

            # Perform factor base maintenance if needed
            maintained = False
            file_paths = glob.glob(os.path.join(pipe_line.factor_save_path, "*.pkl"))
            while len(file_paths) >= 500:
                pipe_line.factor_base_maintain(max_files)
                maintained = True
                file_paths = glob.glob(os.path.join(pipe_line.factor_save_path, "*.pkl"))
            if maintained:
                pipe_line.factor_base_maintain(max_files)
        
        # Final factor base maintenance after completion
        pipe_line.factor_base_maintain(max_files)
