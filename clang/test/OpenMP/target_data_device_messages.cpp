// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

int main(int argc, char **argv) {
  int a, z;
  #pragma omp target data map(to: a) device // expected-error {{expected '(' after 'device'}}
  #pragma omp target data map(to: a) device ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target data map(to: a) device () // expected-error {{expected expression}}
  #pragma omp target data map(to: a) device (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target data map(to: a) device (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target data' are ignored}}
#pragma omp target data map(to: a) device (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp target data map(to: a) device (argc + argc + z)
  #pragma omp target data map(to: a) device (argc), device (argc+1) // expected-error {{directive '#pragma omp target data' cannot contain more than one 'device' clause}}
  #pragma omp target data map(to: a) device (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp target data map(to: a) device (-2) // expected-error {{argument to 'device' clause must be a non-negative integer value}}
  #pragma omp target data map(to: a) device (-10u)
  #pragma omp target data map(to: a) device (ancestor: -10u) // expected-error {{use of undeclared identifier 'ancestor'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target data map(to: a) device (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  foo();

  return 0;
}
