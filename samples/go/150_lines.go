package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Task represents a single to-do item.
type Task struct {
	ID          int
	Title       string
	Description string
	Done        bool
}

// TaskManager manages a list of tasks in memory.
type TaskManager struct {
	tasks []Task
	next  int
}

// NewTaskManager creates a new manager.
func NewTaskManager() *TaskManager {
	return &TaskManager{
		tasks: []Task{},
		next:  1,
	}
}

// AddTask adds a new task to the list.
func (m *TaskManager) AddTask(title, desc string) {
	task := Task{
		ID:          m.next,
		Title:       title,
		Description: desc,
		Done:        false,
	}
	m.tasks = append(m.tasks, task)
	m.next++
	fmt.Println("✅ Task added:", task.Title)
}

// ListTasks shows all tasks.
func (m *TaskManager) ListTasks() {
	if len(m.tasks) == 0 {
		fmt.Println("No tasks yet.")
		return
	}
	fmt.Println("=== Task List ===")
	for _, t := range m.tasks {
		status := "❌"
		if t.Done {
			status = "✔️"
		}
		fmt.Printf("[%s] %d: %s - %s\n", status, t.ID, t.Title, t.Description)
	}
}

// MarkDone marks a task as completed.
func (m *TaskManager) MarkDone(id int) {
	for i := range m.tasks {
		if m.tasks[i].ID == id {
			m.tasks[i].Done = true
			fmt.Println("✔️ Task marked as done:", m.tasks[i].Title)
			return
		}
	}
	fmt.Println("⚠️ Task not found:", id)
}

// DeleteTask removes a task from the list.
func (m *TaskManager) DeleteTask(id int) {
	for i, t := range m.tasks {
		if t.ID == id {
			m.tasks = append(m.tasks[:i], m.tasks[i+1:]...)
			fmt.Println("🗑️ Task deleted:", t.Title)
			return
		}
	}
	fmt.Println("⚠️ Task not found:", id)
}

// CLI loop to interact with user
func main() {
	manager := NewTaskManager()
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Simple Task Manager")
	fmt.Println("===================")
	fmt.Println("Commands: add, list, done, del, help, quit")

	for {
		fmt.Print("> ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		args := strings.Split(input, " ")

		if len(args) == 0 {
			continue
		}

		cmd := args[0]

		switch cmd {
		case "add":
			if len(args) < 3 {
				fmt.Println("Usage: add <title> <description>")
				continue
			}
			title := args[1]
			desc := strings.Join(args[2:], " ")
			manager.AddTask(title, desc)

		case "list":
			manager.ListTasks()

		case "done":
			if len(args) != 2 {
				fmt.Println("Usage: done <id>")
				continue
			}
			id, err := strconv.Atoi(args[1])
			if err != nil {
				fmt.Println("Invalid ID")
				continue
			}
			manager.MarkDone(id)

		case "del":
			if len(args) != 2 {
				fmt.Println("Usage: del <id>")
				continue
			}
			id, err := strconv.Atoi(args[1])
			if err != nil {
				fmt.Println("Invalid ID")
				continue
			}
			manager.DeleteTask(id)

		case "help":
			fmt.Println("Available commands:")
			fmt.Println("  add <title> <desc>   - Add a new task")
			fmt.Println("  list                 - List all tasks")
			fmt.Println("  done <id>            - Mark task as done")
			fmt.Println("  del <id>             - Delete a task")
			fmt.Println("  quit                 - Exit program")

		case "quit":
			fmt.Println("Bye 👋")
			return

		default:
			fmt.Println("Unknown command. Type 'help' for commands.")
		}
	}
}
