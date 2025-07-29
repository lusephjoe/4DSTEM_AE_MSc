#!/usr/bin/env python3
"""Test runner for convert_dm4.py tests with performance reporting."""
import unittest
import sys
import time
from pathlib import Path
from io import StringIO

# Add parent directory to path for importing convert_dm4
sys.path.insert(0, str(Path(__file__).parent.parent))
# Also add current directory for test imports
sys.path.insert(0, str(Path(__file__).parent))

# Import test modules
from test_downsample_strategy import TestDownsampleStrategy, TestDownsampleStrategyIntegration
from test_dm4_converter import (
    TestDM4ConverterInit, TestDM4ConverterMethods, TestDM4ConverterDataTypes,
    TestDM4ConverterValidation, TestDM4ConverterHDF5, TestDM4ConverterIntegration
)
from test_convert_dm4_cli import TestArgumentParser, TestMainFunction, TestCLIEdgeCases


class TimedTestResult(unittest.TextTestResult):
    """Custom test result class that tracks timing."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_times = {}
        self.start_time = None
    
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
    
    def stopTest(self, test):
        super().stopTest(test)
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.test_times[str(test)] = elapsed
    
    def get_slowest_tests(self, n=5):
        """Return the n slowest tests."""
        sorted_tests = sorted(self.test_times.items(), key=lambda x: x[1], reverse=True)
        return sorted_tests[:n]


class TimedTestRunner(unittest.TextTestRunner):
    """Custom test runner that uses TimedTestResult."""
    
    def _makeResult(self):
        return TimedTestResult(self.stream, self.descriptions, self.verbosity)


def create_test_suite():
    """Create comprehensive test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        # Downsampling strategy tests
        TestDownsampleStrategy,
        TestDownsampleStrategyIntegration,
        
        # DM4Converter tests
        TestDM4ConverterInit,
        TestDM4ConverterMethods,
        TestDM4ConverterDataTypes,
        TestDM4ConverterValidation,
        TestDM4ConverterHDF5,
        TestDM4ConverterIntegration,
        
        # CLI tests
        TestArgumentParser,
        TestMainFunction,
        TestCLIEdgeCases,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_specific_test_group(group_name):
    """Run a specific group of tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    groups = {
        'downsample': [TestDownsampleStrategy, TestDownsampleStrategyIntegration],
        'converter': [
            TestDM4ConverterInit, TestDM4ConverterMethods, TestDM4ConverterDataTypes,
            TestDM4ConverterValidation, TestDM4ConverterHDF5, TestDM4ConverterIntegration
        ],
        'cli': [TestArgumentParser, TestMainFunction, TestCLIEdgeCases],
    }
    
    if group_name not in groups:
        print(f"Unknown test group: {group_name}")
        print(f"Available groups: {list(groups.keys())}")
        return False
    
    for test_class in groups[group_name]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = TimedTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_fast_tests_only():
    """Run only fast tests (exclude integration tests)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Only include unit tests, exclude integration tests
    fast_test_classes = [
        TestDownsampleStrategy,  # Keep strategy tests
        TestDM4ConverterInit,
        TestDM4ConverterMethods,
        TestDM4ConverterDataTypes,
        TestDM4ConverterValidation,
        TestArgumentParser,
        TestCLIEdgeCases,
    ]
    
    for test_class in fast_test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = TimedTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def print_test_summary(result):
    """Print a summary of test results."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if hasattr(result, 'test_times'):
        print(f"\nSlowest tests:")
        for test_name, duration in result.get_slowest_tests():
            print(f"  {test_name}: {duration:.3f}s")
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if failures > 0:
        print(f"\nFAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if errors > 0:
        print(f"\nERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"  - {test}")


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test runner for convert_dm4.py")
    parser.add_argument("--group", choices=['downsample', 'converter', 'cli'],
                       help="Run specific test group")
    parser.add_argument("--fast", action="store_true",
                       help="Run only fast tests (no integration tests)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--failfast", action="store_true",
                       help="Stop on first failure")
    
    args = parser.parse_args()
    
    print("Convert DM4 Test Suite")
    print("="*40)
    
    # Configure test runner
    verbosity = 2 if args.verbose else 1
    
    if args.group:
        print(f"Running {args.group} tests only...")
        success = run_specific_test_group(args.group)
    elif args.fast:
        print("Running fast tests only (excluding integration tests)...")
        suite = create_test_suite() if not args.fast else None
        
        if args.fast:
            success = run_fast_tests_only()
        else:
            runner = TimedTestRunner(verbosity=verbosity, failfast=args.failfast)
            result = runner.run(suite)
            print_test_summary(result)
            success = result.wasSuccessful()
    else:
        print("Running all tests...")
        suite = create_test_suite()
        runner = TimedTestRunner(verbosity=verbosity, failfast=args.failfast)
        result = runner.run(suite)
        print_test_summary(result)
        success = result.wasSuccessful()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()