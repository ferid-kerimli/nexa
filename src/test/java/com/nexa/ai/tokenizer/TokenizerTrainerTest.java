package com.nexa.ai.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerTrainerTest {

    @Test
    void shouldBuildVocabularyFromTexts() {

        TokenizerTrainer trainer = new TokenizerTrainer();

        Vocabulary vocabulary = trainer.train(
                List.of(
                        "Java is powerful",
                        "Java is fast",
                        "Java is popular"
                )
        );

        assertTrue(vocabulary.contains("Java"));
        assertTrue(vocabulary.contains("is"));
        assertTrue(vocabulary.contains("powerful"));
        assertTrue(vocabulary.contains("fast"));
        assertTrue(vocabulary.contains("popular"));
    }

    @Test
    void shouldAssignMostFrequentTokenFirst() {

        TokenizerTrainer trainer = new TokenizerTrainer();

        Vocabulary vocabulary = trainer.train(
                List.of(
                        "Java Java Java",
                        "Python Python",
                        "Rust"
                )
        );

        assertEquals(2, vocabulary.getId("Java"));
        assertEquals(3, vocabulary.getId("Python"));
        assertEquals(4, vocabulary.getId("Rust"));
    }

    @Test
    void shouldIgnoreBlankTexts() {

        TokenizerTrainer trainer = new TokenizerTrainer();

        Vocabulary vocabulary = trainer.train(
                List.of(
                        "",
                        "   ",
                        "Java"
                )
        );

        assertTrue(vocabulary.contains("Java"));
        assertEquals(3, vocabulary.size());
    }

    @Test
    void shouldKeepSpecialTokens() {

        TokenizerTrainer trainer = new TokenizerTrainer();

        Vocabulary vocabulary = trainer.train(
                List.of("Java is powerful")
        );

        assertEquals(0, vocabulary.getId(Vocabulary.UNK_TOKEN));
        assertEquals(1, vocabulary.getId(Vocabulary.PAD_TOKEN));
    }
}