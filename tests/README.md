# Tests

This directory contains test suites for the AI Security Labs project.

## Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_common.py           # Tests for common utilities
└── ml01/
    └── test_ml01.py         # Tests for ML01 lab components
```

## Running Tests

Run all tests:
```bash
make test
```

Run specific test module:
```bash
make test-unit
```

Run with coverage:
```bash
make test-coverage
```

## Test Categories

### Unit Tests
- Model architecture tests
- Attack algorithm tests  
- Defense mechanism tests
- Utility function tests

### Integration Tests
- Complete attack-defense pipeline
- End-to-end workflow validation
- Cross-component interaction tests

### Performance Tests
- Attack success rate validation
- Defense detection accuracy
- Computational efficiency checks

## Writing Tests

When adding new tests:

1. Use descriptive test names
2. Include docstrings explaining test purpose
3. Use fixtures for setup/teardown
4. Test both success and error cases
5. Include edge case validation
6. Mock external dependencies when appropriate

## Test Coverage

Current test coverage includes:
- SimpleCNN model functionality
- FGSM attack implementation
- Adversarial defense mechanisms
- Utility functions and logging
- Error handling and validation
