from aahar.scenarios import reply


PROFILE_ALLERGENS = ["Peanuts", "Milk", "Eggs", "Crustaceans", "Wheat"]


def test_restaurant_answers_use_the_exact_dish_and_selected_level():
    beginner = reply(
        "restaurant_beginner",
        "What ingredients are in the satay chicken skewers?",
        PROFILE_ALLERGENS,
        0,
    )
    assert "Satay Chicken Skewers" in beginner["text"]
    assert "Declared allergens: peanuts, soybeans, sesame" in beginner["text"]
    assert "Prawn Tempura" not in beginner["text"]

    advanced = reply(
        "restaurant_advanced",
        "What allergens are in the mini chicken Caesar bites?",
        PROFILE_ALLERGENS,
        0,
    )
    assert "Mini Chicken Caesar Bites" in advanced["text"]
    assert "milk, egg" in advanced["text"]


def test_does_not_substitute_a_similar_dish_or_other_level_menu():
    result = reply(
        "restaurant_intermediate",
        "Does the prawn tempura contain egg?",
        PROFILE_ALLERGENS,
        0,
    )
    assert "Prawn Tempura" in result["text"]
    assert "on this level's training menu" in result["text"]
    assert "Prawn Cocktail" not in result["text"]


def test_follow_up_menu_question_resolves_previous_dish():
    result = reply(
        "restaurant_intermediate",
        "Does it contain peanuts?",
        PROFILE_ALLERGENS,
        1,
        ["Can you check the Chicken Satay Skewers ingredients?"],
    )
    assert "Chicken Satay Skewers" in result["text"]
    assert "Declared allergens: peanuts, soybeans" in result["text"]


def test_allergen_question_is_not_scored_as_disclosure():
    question = reply(
        "restaurant_beginner",
        "Does the satay contain peanuts?",
        PROFILE_ALLERGENS,
        0,
    )
    disclosure = reply(
        "restaurant_beginner",
        "I am allergic to peanuts.",
        PROFILE_ALLERGENS,
        0,
    )
    assert question["points"] == 0
    assert disclosure["points"] == 10
    assert "Thanks for telling me" in disclosure["text"]
    assert reply("restaurant_beginner", "I have a peanut allergy.", PROFILE_ALLERGENS, 0)["points"] == 10


def test_generic_order_request_returns_cautious_personalized_options():
    result = reply("restaurant_beginner", "What can I order?", ["Peanuts"], 0)
    assert "do not list Peanuts" in result["text"]
    assert "Tomato & Basil Soup" in result["text"]
    assert "not a safety guarantee" in result["text"]
    assert "Before choosing" in result["coach"]

    no_profile = reply("restaurant_beginner", "Could you recommend something?", [], 0)
    assert "don't have your allergies yet" in no_profile["text"]


def test_group_scenario_admits_unknown_cake_ingredients_instead_of_guessing():
    result = reply(
        "party_intermediate",
        "Does the birthday cake contain peanuts?",
        PROFILE_ALLERGENS,
        0,
    )
    assert "not sure what's in that food" in result["text"]
    assert "check the label or ask the host" in result["text"]
