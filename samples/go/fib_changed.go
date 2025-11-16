package main

import "fmt"

// fib returns the n-th Fibonacci number using iterative DP.
func fiboo(n int) int {
	if n < 2 {
		return n
	}
	a, b := 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

func main() {
	for i := 0; i < 10; i++ {
		fmt.Printf("fib(%d) = %d\n", i, fib(i))
	}
}