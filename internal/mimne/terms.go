package mimne

import (
	"regexp"
	"strings"
)

var stopwords = map[string]bool{
	"a": true, "an": true, "the": true, "some": true, "any": true,
	"this": true, "that": true, "these": true, "those": true,
	"my": true, "your": true, "our": true, "its": true, "their": true,
	"i": true, "me": true, "we": true, "you": true, "he": true,
	"she": true, "it": true, "they": true, "them": true,
	"is": true, "am": true, "are": true, "was": true, "were": true,
	"be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true,
	"do": true, "does": true, "did": true,
	"will": true, "would": true, "could": true, "should": true,
	"can": true, "may": true, "might": true, "shall": true, "must": true,
	"to": true, "of": true, "in": true, "for": true, "on": true,
	"with": true, "at": true, "by": true, "from": true, "as": true,
	"into": true, "about": true, "but": true, "or": true, "and": true,
	"not": true, "no": true, "if": true, "so": true,
	"than": true, "too": true, "very": true, "just": true, "also": true,
	"then": true, "now": true, "here": true,
	"there": true, "when": true, "where": true, "how": true,
	"what": true, "which": true, "who": true, "whom": true,
	"up": true, "out": true, "all": true, "each": true, "every": true,
	"both": true, "few": true, "more": true,
	"other": true, "such": true, "only": true, "own": true,
	"same": true, "still": true, "after": true, "before": true,
	"over": true, "under": true, "between": true, "through": true,
	"during": true, "above": true, "below": true,
	"again": true, "once": true, "much": true, "many": true,
	"well": true, "back": true, "even": true,
	"get": true, "got": true, "going": true, "go": true,
	"want": true, "need": true, "know": true, "think": true,
	"like": true, "make": true, "take": true, "come": true,
	"see": true, "look": true, "give": true,
	"us": true, "him": true, "her": true,
	"something": true, "anything": true, "nothing": true,
	"everything": true, "thing": true, "things": true,
	"really": true, "pretty": true, "quite": true,
	"let": true, "say": true, "said": true, "tell": true, "told": true,
	"yes": true, "no ": true, "ok": true, "okay": true, "sure": true,
	"right": true, "don": true, "doesn": true, "didn": true,
	"won": true, "isn": true, "aren": true, "wasn": true, "weren": true,
	"hasn": true, "haven": true, "hadn": true,
	"shouldn": true, "couldn": true, "wouldn": true,
	"ve": true, "re": true, "ll": true, "d": true, "s": true, "t": true,
}

var wordRe = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9_'\-]*[a-zA-Z0-9]|[a-zA-Z]`)

var intentPatterns = map[string][]*regexp.Regexp{
	"temporal": {
		regexp.MustCompile(`(?i)\b(recent|latest|last\s+\w+|yesterday|today|this\s+week|since|changed|updates?|what\s+did\s+we\s+do)\b`),
	},
	"causal": {
		regexp.MustCompile(`(?i)\b(why\s+did|why\s+do|why\s+is|what\s+caused|reason\s+for|how\s+come|what\s+went\s+wrong|what\s+broke)\b`),
	},
	"procedural": {
		regexp.MustCompile(`(?i)\b(how\s+do\s+[iwe]|how\s+to|steps?\s+to|process\s+for|what'?s\s+the\s+command|procedure|teardown|deploy|run\s+the)\b`),
	},
	"definitional": {
		regexp.MustCompile(`(?i)\b(what\s+is|what\s+are|tell\s+me\s+about|describe|explain|define|overview|summary\s+of)\b`),
	},
}

var executionPatterns = []*regexp.Regexp{
	regexp.MustCompile("```"),
	regexp.MustCompile(`[~/][a-zA-Z0-9_\-./]+`),
	regexp.MustCompile(`(?m)^\$\s`),
	regexp.MustCompile(`(?m)^>>>\s`),
	regexp.MustCompile(`(?i)\b(deployed|created|deleted|applied|terraform|cloudformation)\b`),
	regexp.MustCompile(`\b(Error|Success|FAILED|PASSED|status.code)\b`),
	regexp.MustCompile(`(?i)\b(wrote to|edited|read file|executed)\b`),
}

// ClassifyIntent classifies a user message's intent for retrieval routing.
func ClassifyIntent(message string) string {
	for intent, patterns := range intentPatterns {
		for _, p := range patterns {
			if p.MatchString(message) {
				return intent
			}
		}
	}
	return "default"
}

// ExtractSearchTerms extracts search terms using lightweight regex.
// Keeps capitalized words as proper nouns, non-stopword tokens, and bigrams.
func ExtractSearchTerms(message string) []string {
	words := wordRe.FindAllString(message, -1)
	var terms []string
	seen := make(map[string]bool)

	// Pass 1: capitalized words (proper nouns)
	for _, w := range words {
		if len(w) > 1 && w[0] >= 'A' && w[0] <= 'Z' {
			low := strings.ToLower(w)
			if !stopwords[low] && !seen[low] {
				terms = append(terms, low)
				seen[low] = true
			}
		}
	}

	// Pass 2: all non-stopword tokens
	var survivors []string
	for _, w := range words {
		low := strings.ToLower(w)
		if !stopwords[low] && len(low) > 1 {
			if !seen[low] {
				terms = append(terms, low)
				seen[low] = true
			}
			survivors = append(survivors, low)
		}
	}

	// Pass 3: bigrams from adjacent survivors
	for i := 0; i < len(survivors)-1; i++ {
		bigram := survivors[i] + " " + survivors[i+1]
		if !seen[bigram] {
			terms = append(terms, bigram)
			seen[bigram] = true
		}
	}

	return terms
}

// BuildTSQueryString builds a PostgreSQL tsquery string from extracted terms.
// Multi-word terms use phrase matching (<->).
func BuildTSQueryString(terms []string) string {
	sanitizeRe := regexp.MustCompile(`[^\w]`)
	var parts []string

	for _, term := range terms {
		words := strings.Fields(strings.TrimSpace(term))
		var sanitized []string
		for _, w := range words {
			s := sanitizeRe.ReplaceAllString(w, "")
			if s != "" {
				sanitized = append(sanitized, s)
			}
		}
		if len(sanitized) == 0 {
			continue
		}
		if len(sanitized) == 1 {
			parts = append(parts, sanitized[0])
		} else {
			phrase := "(" + strings.Join(sanitized, " <-> ") + ")"
			parts = append(parts, phrase)
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " | ")
}

// HasExecutionSignal checks if text contains evidence of execution/action.
func HasExecutionSignal(text string) bool {
	for _, p := range executionPatterns {
		if p.MatchString(text) {
			return true
		}
	}
	return false
}
