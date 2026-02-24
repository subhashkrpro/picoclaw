package agent

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestSystemPromptCaching verifies that BuildSystemPrompt caches results
func TestSystemPromptCaching(t *testing.T) {
	// Create a temporary workspace
	tmpDir, err := os.MkdirTemp("", "test-workspace")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create memory directory
	memDir := filepath.Join(tmpDir, "memory")
	os.MkdirAll(memDir, 0o755)

	// Create context builder
	cb := NewContextBuilder(tmpDir)

	// First call should build and cache
	prompt1 := cb.BuildSystemPrompt()
	if prompt1 == "" {
		t.Fatal("First call to BuildSystemPrompt returned empty string")
	}

	// Second call should return cached value (identical string)
	prompt2 := cb.BuildSystemPrompt()
	if prompt1 != prompt2 {
		t.Errorf("Cached prompts differ:\nFirst: %s\nSecond: %s", prompt1, prompt2)
	}

	// Verify it's using cache by checking time
	cb.systemPromptCacheMu.RLock()
	cacheTime := cb.systemPromptTime
	cb.systemPromptCacheMu.RUnlock()

	// Wait a bit and verify cache is still fresh
	time.Sleep(100 * time.Millisecond)
	prompt3 := cb.BuildSystemPrompt()
	if prompt1 != prompt3 {
		t.Error("Cache was invalidated prematurely")
	}

	// Verify cache time hasn't changed (still using cache)
	cb.systemPromptCacheMu.RLock()
	newCacheTime := cb.systemPromptTime
	cb.systemPromptCacheMu.RUnlock()
	if !cacheTime.Equal(newCacheTime) {
		t.Error("Cache was rebuilt instead of reused")
	}
}

// TestSystemPromptCacheInvalidation verifies cache can be invalidated
func TestSystemPromptCacheInvalidation(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "test-workspace")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	memDir := filepath.Join(tmpDir, "memory")
	os.MkdirAll(memDir, 0o755)

	cb := NewContextBuilder(tmpDir)

	// Build and cache
	prompt1 := cb.BuildSystemPrompt()

	// Verify it's cached
	cb.systemPromptCacheMu.RLock()
	cacheTime1 := cb.systemPromptTime
	cb.systemPromptCacheMu.RUnlock()

	// Wait a bit
	time.Sleep(50 * time.Millisecond)

	// Invalidate cache
	cb.InvalidateSystemPromptCache()

	// Next call should rebuild
	prompt2 := cb.BuildSystemPrompt()

	// Verify cache was updated (newer timestamp)
	cb.systemPromptCacheMu.RLock()
	cacheTime2 := cb.systemPromptTime
	cb.systemPromptCacheMu.RUnlock()

	if !cacheTime2.After(cacheTime1) {
		t.Error("Cache invalidation didn't update cache time")
	}

	// Content should be the same (if files haven't changed)
	if prompt1 != prompt2 {
		t.Error("Prompt content changed after cache invalidation (unexpected)")
	}
}

// TestSystemPromptCacheExpiration verifies cache expires after TTL
func TestSystemPromptCacheExpiration(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "test-workspace")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	memDir := filepath.Join(tmpDir, "memory")
	os.MkdirAll(memDir, 0o755)

	cb := NewContextBuilder(tmpDir)

	// Set a very short TTL for testing
	cb.cacheTTL = 100 * time.Millisecond

	// Build and cache
	prompt1 := cb.BuildSystemPrompt()

	// Get cache time
	cb.systemPromptCacheMu.RLock()
	cacheTime1 := cb.systemPromptTime
	cb.systemPromptCacheMu.RUnlock()

	// Wait for cache to expire
	time.Sleep(150 * time.Millisecond)

	// Next call should rebuild
	prompt2 := cb.BuildSystemPrompt()

	// Verify cache was updated (newer timestamp)
	cb.systemPromptCacheMu.RLock()
	cacheTime2 := cb.systemPromptTime
	cb.systemPromptCacheMu.RUnlock()

	if !cacheTime2.After(cacheTime1) {
		t.Error("Cache wasn't expired and rebuilt")
	}

	// Content should be the same
	if prompt1 != prompt2 {
		t.Error("Prompt content differs after cache expiration")
	}
}

// TestForceCompressionDoesNotCorruptHistory verifies the fixed forceCompression works
func TestForceCompressionDoesNotCorruptHistory(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "test-workspace")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create mock agent
	agent := &AgentInstance{
		ID:             "test",
		Workspace:      tmpDir,
		Sessions:       NewSessionManager(filepath.Join(tmpDir, "sessions")),
		ContextBuilder: NewContextBuilder(tmpDir),
	}

	sessionKey := "test-session"

	// Create history with multiple messages
	history := []interface{}{
		map[string]string{"role": "user", "content": "message 1"},
		map[string]string{"role": "assistant", "content": "response 1"},
		map[string]string{"role": "user", "content": "message 2"},
		map[string]string{"role": "assistant", "content": "response 2"},
		map[string]string{"role": "user", "content": "message 3"},
		map[string]string{"role": "assistant", "content": "response 3"},
	}

	// Add messages to session
	for _, msg := range history {
		m := msg.(map[string]string)
		agent.Sessions.AddMessage(sessionKey, m["role"], m["content"])
	}

	// Verify original history
	origHistory := agent.Sessions.GetHistory(sessionKey)
	if len(origHistory) != 6 {
		t.Fatalf("Expected 6 messages, got %d", len(origHistory))
	}

	// Create agent loop for compression
	al := &AgentLoop{state: NewManager(tmpDir)}

	// Force compression
	al.forceCompression(agent, sessionKey)

	// Verify history is compressed
	compressedHistory := agent.Sessions.GetHistory(sessionKey)
	if len(compressedHistory) >= len(origHistory) {
		t.Errorf("Expected history to be compressed, got %d messages (was %d)",
			len(compressedHistory), len(origHistory))
	}

	// Verify no message is a system message with "Emergency compression" note
	for i, msg := range compressedHistory {
		if msg.Role == "system" {
			t.Errorf("Found unexpected system message at index %d: %s", i, msg.Content)
		}
		if contains(msg.Content, "Emergency compression") {
			t.Errorf("Found compression note in message at index %d: %s", i, msg.Content)
		}
	}

	// Verify messages are from the original history (not corrupted)
	for i, msg := range compressedHistory {
		orig := origHistory[len(origHistory)-len(compressedHistory)+i]
		if msg.Content != orig.Content {
			t.Errorf("Message %d corrupted: expected %q, got %q", i, orig.Content, msg.Content)
		}
	}
}

func contains(s, substr string) bool {
	return len(substr) > 0 && len(s) >= len(substr) &&
		(s == substr || len(substr) > 0 && s[:len(substr)] == substr ||
			len(s) >= len(substr) && contains(s[1:], substr))
}

// Simplified contains function for substring search
func stringContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestForceCompressionDoesNotCorruptHistoryV2(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "test-workspace")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create mock agent
	agent := &AgentInstance{
		ID:             "test",
		Workspace:      tmpDir,
		Sessions:       NewSessionManager(filepath.Join(tmpDir, "sessions")),
		ContextBuilder: NewContextBuilder(tmpDir),
	}

	sessionKey := "test-session"

	// Create history with multiple messages
	messages := []struct {
		role    string
		content string
	}{
		{"user", "message 1"},
		{"assistant", "response 1"},
		{"user", "message 2"},
		{"assistant", "response 2"},
		{"user", "message 3"},
		{"assistant", "response 3"},
	}

	// Add messages to session
	for _, msg := range messages {
		agent.Sessions.AddMessage(sessionKey, msg.role, msg.content)
	}

	// Verify original history
	origHistory := agent.Sessions.GetHistory(sessionKey)
	if len(origHistory) != 6 {
		t.Fatalf("Expected 6 messages, got %d", len(origHistory))
	}

	// Create agent loop for compression
	al := &AgentLoop{state: NewManager(tmpDir)}

	// Force compression
	al.forceCompression(agent, sessionKey)

	// Verify history is compressed
	compressedHistory := agent.Sessions.GetHistory(sessionKey)
	t.Logf("Original history length: %d, Compressed length: %d", len(origHistory), len(compressedHistory))

	if len(compressedHistory) >= len(origHistory) {
		t.Errorf("Expected history to be compressed, got %d messages (was %d)",
			len(compressedHistory), len(origHistory))
	}

	// Verify no message is a system message with "Emergency compression" note
	for i, msg := range compressedHistory {
		if msg.Role == "system" {
			t.Errorf("Found unexpected system message at index %d: %s", i, msg.Content)
		}
		if stringContains(msg.Content, "Emergency compression") {
			t.Errorf("Found compression note in message at index %d: %s", i, msg.Content)
		}
	}

	// Verify messages are kept from the end (should be most recent messages)
	numDropped := len(origHistory) - len(compressedHistory)
	t.Logf("Kept messages from end: %d messages (dropped %d from start)", len(compressedHistory), numDropped)

	// The kept messages should match the last N messages from original
	for i, msg := range compressedHistory {
		origIdx := numDropped + i
		if origIdx < len(origHistory) {
			orig := origHistory[origIdx]
			if msg.Content != orig.Content || msg.Role != orig.Role {
				t.Errorf("Message mismatch at position %d: expected {%s: %q}, got {%s: %q}",
					i, orig.Role, orig.Content, msg.Role, msg.Content)
			}
		}
	}
}
