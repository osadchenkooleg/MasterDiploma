func echo(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var in map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	out := resp{ReceivedAt: time.Now().UTC(), Payload: in}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}
