TOY_PROBLEMS = [
    (
        "What's 5/4 + 2?",
        "5/4 is 1, so",  # Wrong fraction conversion, unfinished
        "Let me solve this step by step:\n1. 5/4 = 1.25\n2. 1.25 + 2 = 3.25\nTherefore, 5/4 + 2 = 3.25"
    ),
    (
        "If a train travels 120 miles in 2 hours, what's its speed in miles per hour?",
        "Let me divide 2 by 120, so",  # Wrong division order, unfinished
        "Let me solve this step by step:\n1. Speed = Distance ÷ Time\n2. Distance = 120 miles\n3. Time = 2 hours\n4. Speed = 120 ÷ 2 = 60\nTherefore, the train's speed is 60 miles per hour"
    ),
    (
        "Sally has 3 times as many marbles as Tom. If Tom has 12 marbles, how many do they have together?",
        "If Tom has 12 marbles, Sally has 3 marbles, so",  # Misinterpreted 'times as many', unfinished
        "Let me solve this step by step:\n1. Tom has 12 marbles\n2. Sally has 3 × 12 = 36 marbles\n3. Together they have 12 + 36 = 48 marbles\nTherefore, Sally and Tom have 48 marbles together"
    ),
    (
        "What's 2 + 3 × 4?",
        "First I'll add 2 and 3 to get 5, then",  # Order of operations mistake, unfinished
        "Let me solve this step by step:\n1. Following PEMDAS, multiply first: 3 × 4 = 12\n2. Then add: 2 + 12 = 14\nTherefore, 2 + 3 × 4 = 14"
    ),
    (
        "A rectangle has a width of 4 inches and a length twice its width. What's its area?",
        "If the width is 4 inches, then the length is 4 + 2 = 6 inches. Now to find the area,",  # Wrong interpretation of 'twice'
        "Let me solve this step by step:\n1. Width = 4 inches\n2. Length = 2 × width = 2 × 4 = 8 inches\n3. Area = length × width = 8 × 4 = 32\nTherefore, the rectangle's area is 32 square inches"
    ),
    (
        "If you have $20 and spend 25% of it, how much do you have left?",
        "25% of $20 is $5, so I'll add $5 to get",  # Wrong operation (adding instead of subtracting)
        "Let me solve this step by step:\n1. 25% of $20 = $20 × 0.25 = $5\n2. Amount left = $20 - $5 = $15\nTherefore, you have $15 left"
    ),
    (
        "What's the average of 15, 20, and 25?",
        "To find the average, I'll add these numbers: 15 + 20 + 25 = 50. Now I'll divide by 2 since",  # Wrong divisor
        "Let me solve this step by step:\n1. Sum the numbers: 15 + 20 + 25 = 60\n2. Count the numbers: 3\n3. Divide sum by count: 60 ÷ 3 = 20\nTherefore, the average is 20"
    ),
    (
        "If 8 cookies are shared equally among 4 children, how many cookies does each child get?",
        "I'll multiply 8 × 4 to find out how many cookies each child gets, so",  # Wrong operation
        "Let me solve this step by step:\n1. Total cookies = 8\n2. Number of children = 4\n3. Cookies per child = 8 ÷ 4 = 2\nTherefore, each child gets 2 cookies"
    ),
    (
        "What's 1/2 of 30?",
        "To find half of 30, I'll add 30 + 2, which gives me",  # Wrong operation
        "Let me solve this step by step:\n1. 1/2 means dividing by 2\n2. 30 ÷ 2 = 15\nTherefore, 1/2 of 30 is 15"
    ),
    (
        "A square has a perimeter of 20 inches. What's its area?",
        "If the perimeter is 20 inches, each side must be 20/2 = 10 inches. Now for the area,",  # Wrong perimeter calculation
        "Let me solve this step by step:\n1. Perimeter = 4 × side length\n2. 20 = 4 × side length\n3. Side length = 20 ÷ 4 = 5 inches\n4. Area = side length² = 5 × 5 = 25\nTherefore, the area is 25 square inches"
    ),
    (
        "How many quarters make $2.75?",
        "Each quarter is 25 cents, which is $0.25. So I'll multiply 2.75 × 0.25 to get",  # Wrong approach
        "Let me solve this step by step:\n1. Convert $2.75 to cents: $2.75 × 100 = 275 cents\n2. Each quarter is 25 cents\n3. Number of quarters = 275 ÷ 25 = 11\nTherefore, 11 quarters make $2.75"
    ),
    (
        "If it takes 3 minutes to boil one egg, how long will it take to boil 6 eggs at the same time?",
        "With 6 eggs, it will take 6 × 3 = 18 minutes because",  # Wrong reasoning about parallel vs sequential
        "Let me solve this step by step:\n1. All eggs can be boiled simultaneously in the same pot\n2. Time to boil one egg = 3 minutes\n3. Time to boil multiple eggs simultaneously = 3 minutes\nTherefore, it will take 3 minutes to boil 6 eggs at the same time"
    ),
    (
        "Three friends split a pizza bill of $45. If Tom pays $5 more than Jack, and Jack pays $3 more than Mike, how much did each person pay?",
        "Let me start by dividing $45 by 3 to get each person's share, which is $15. Now,",  # Wrong approach
        "Let me solve this step by step:\n1. Let Mike's payment be x\n2. Then Jack pays (x + 3) and Tom pays (x + 8)\n3. Total bill: x + (x + 3) + (x + 8) = $45\n4. 3x + 11 = 45\n5. 3x = 34\n6. x = $11.33 (rounded to nearest cent)\n7. Jack pays: $11.33 + $3 = $14.33\n8. Tom pays: $11.33 + $8 = $19.33\nTherefore, Mike pays $11.33, Jack pays $14.33, and Tom pays $19.33"
    ),
    (
        "A store offers a 20% discount on a $80 jacket, then charges 8% sales tax. What's the final price?",
        "First I'll add the 8% tax to $80, which is $86.40, then apply the 20% discount",  # Wrong order of operations
        "Let me solve this step by step:\n1. Original price = $80\n2. 20% discount = $80 × 0.20 = $16\n3. Price after discount = $80 - $16 = $64\n4. 8% tax on discounted price = $64 × 0.08 = $5.12\n5. Final price = $64 + $5.12 = $69.12\nTherefore, the final price is $69.12"
    ),
    (
        "In a class of 30 students, 60% play sports and 40% play music. If 5 students play both, how many students don't participate in either activity?",
        "60% of 30 is 18 students playing sports, 40% is 12 playing music. So 18 + 12 = 30 students total, meaning",  # Failed to account for overlap
        "Let me solve this step by step:\n1. Students playing sports = 60% of 30 = 18 students\n2. Students playing music = 40% of 30 = 12 students\n3. Students playing both = 5\n4. Total students in activities = 18 + 12 - 5 = 25 (subtract overlap)\n5. Students in neither = 30 - 25 = 5\nTherefore, 5 students don't participate in either activity"
    ),
    (
        "A box contains red, blue, and green marbles. If 1/3 are red, 1/4 are blue, what fraction are green?",
        "Let me add 1/3 + 1/4 to get 8/12 for red and blue combined. So green must be 4/12 because",  # Wrong fraction addition
        "Let me solve this step by step:\n1. Convert to common denominator of 12:\n   - Red: 1/3 = 4/12\n   - Blue: 1/4 = 3/12\n2. Red and blue together = 4/12 + 3/12 = 7/12\n3. Total must be 12/12 (or 1)\n4. Green = 12/12 - 7/12 = 5/12\nTherefore, 5/12 of the marbles are green"
    ),
    (
        "Five people ordered different dishes at a restaurant: pasta, steak, salad, fish, and soup. We know that:\n- Alice sat between Bob and Carol\n- The person who ordered steak sat next to the person who ordered salad\n- Dave ordered soup and sat at one end\n- Eve sat next to the person who ordered pasta\n- Bob didn't order fish\nWhat did each person order?",
        "Since Dave ordered soup and sat at one end, and Alice is between Bob and Carol, Alice must be next to Dave. So Alice ordered steak and",  # Jumping to conclusions without using all constraints
        "Let me solve this step by step:\n1. Dave has soup and sits at one end\n2. Alice is between Bob and Carol\n3. Eve sits next to pasta person\n4. Bob can't have fish\n5. Working through seating: Dave-Bob-Alice-Carol-Eve or Eve-Carol-Alice-Bob-Dave\n6. Given Eve next to pasta and steak next to salad:\n   - Dave: soup\n   - Bob: steak\n   - Alice: salad\n   - Carol: pasta\n   - Eve: fish\nTherefore, Dave has soup, Bob has steak, Alice has salad, Carol has pasta, and Eve has fish"
    ),
    (
        "A train leaves Station A at 2:00 PM traveling 60 mph. Another train leaves Station B at 2:30 PM traveling 75 mph in the opposite direction. If the stations are 300 miles apart, at what time do the trains meet?",
        "The trains are traveling at a combined speed of 60 + 75 = 135 mph. Since they need to cover 300 miles, I'll divide 300 by 135 to get",  # Forgot to account for 30-minute delay
        "Let me solve this step by step:\n1. Combined speed = 60 + 75 = 135 mph\n2. In first 30 minutes, Train A travels: 60 mph × 0.5 hours = 30 miles\n3. Remaining distance = 300 - 30 = 270 miles\n4. Time to cover remaining distance = 270 ÷ 135 = 2 hours\n5. Total time from Train B's departure = 2 hours\n6. Train B departed at 2:30 PM + 2 hours = 4:30 PM\nTherefore, the trains meet at 4:30 PM"
    ),
    (
        "A factory produces widgets in three colors: red, blue, and green. They made 1200 widgets last month. If red widgets take 2 hours to make, blue widgets take 3 hours, and green widgets take 4 hours, and the factory spent exactly 3000 hours on production, how many of each color did they make?",
        "Let's solve this step by step:\n1. Let's say we have r red, b blue, and g green widgets\n2. We know r + b + g = 1200\n3. And 2r + 3b + 4g = 3000\n4. Since red widgets are fastest, let's maximize those first, so",  # Error in step 4 - wrong approach
        "Let me solve this step by step:\n1. Let r, b, g be the number of red, blue, green widgets\n2. Equation 1: r + b + g = 1200 (total widgets)\n3. Equation 2: 2r + 3b + 4g = 3000 (total hours)\n4. From Equation 1: r = 1200 - b - g\n5. Substitute into Equation 2:\n   2(1200 - b - g) + 3b + 4g = 3000\n   2400 - 2b - 2g + 3b + 4g = 3000\n   2400 + b + 2g = 3000\n   b + 2g = 600\n6. Solve system:\n   b = 600 - 2g\n   r = 1200 - (600 - 2g) - g = 600 + g\n7. All values must be positive, solving gives:\n   g = 200, b = 200, r = 800\nTherefore, they made 800 red, 200 blue, and 200 green widgets"
    ),
    (
        "A baker makes cakes that require 3 eggs, 2 cups of flour, and 1 cup of sugar each. Cookies require 1 egg, 1 cup of flour, and 1/2 cup of sugar each. If the baker has 24 eggs, 20 cups of flour, and 12 cups of sugar, and wants to maximize the total number of items made, how many of each should they bake?",
        "Let's solve this:\n1. Let's say we make x cakes and y cookies\n2. For eggs: 3x + y ≤ 24\n3. For flour: 2x + y ≤ 20\n4. For sugar: x + 0.5y ≤ 12\n5. Since cakes use more resources, we should minimize cakes and maximize cookies, so",  # Error in step 5
        "Let me solve this step by step:\n1. Let x be cakes and y be cookies\n2. Constraints:\n   - Eggs: 3x + y ≤ 24\n   - Flour: 2x + y ≤ 20\n   - Sugar: x + 0.5y ≤ 12\n3. To maximize items (x + y), test each constraint:\n   - From eggs: y = 24 - 3x\n   - From flour: y = 20 - 2x\n   - From sugar: y = 24 - 2x\n4. The flour constraint is most limiting\n5. Substitute y = 20 - 2x into x + y to maximize:\n   x + (20 - 2x) = 20 - x\n6. Since we want to maximize items, and the function decreases with x,\n   x = 4 (minimum value that satisfies all constraints)\n7. Then y = 12\nTherefore, bake 4 cakes and 12 cookies for a total of 16 items"
    ),
    (
        "Three teams (Red, Blue, Yellow) compete in a round-robin tournament where each team plays every other team twice. Points are awarded as follows: 3 for a win, 1 for a tie, 0 for a loss. After all games are played:\n- Red team scored 7 points\n- Blue team scored 4 points\n- Yellow team scored 7 points\nHow many games did each team win, lose, and tie?",
        "Let's solve this:\n1. In total, there are 6 games played (each team plays each other twice)\n2. Total points awarded is 18 (7 + 4 + 7)\n3. Since Red and Yellow have equal points, they must have identical records\n4. A win must be balanced by a loss, so there must be equal wins and losses",  # Error in step 4
        "Let me solve this step by step:\n1. Total games = 6 (each team plays others twice)\n2. Red and Yellow both have 7 points, Blue has 4\n3. Each game must have one winner and one loser (or two ties)\n4. For Red and Yellow to have equal points (7):\n   - They must have tied their games against each other\n   - Each must have won twice against Blue\n5. Therefore:\n   Red: 2 wins (vs Blue), 2 ties (vs Yellow), 2 losses (vs Blue)\n   Yellow: 2 wins (vs Blue), 2 ties (vs Red), 2 losses (vs Blue)\n   Blue: 2 wins (vs Red & Yellow), 0 ties, 4 losses (vs Red & Yellow)\nTherefore, Red and Yellow each had 2 wins, 2 ties, and 2 losses; Blue had 2 wins, 0 ties, and 4 losses"
    ),
    (
        "A store sells paint in 3 sizes: small ($10), medium ($18), and large ($25). During a sale, if you buy any 2 cans, you get 30% off the lower priced can. If you buy 3 cans, you get 40% off the lowest priced can. What's the best price for buying 2 small and 1 medium can?",
        "Let's calculate: Two small cans at $10 each is $20, plus $18 for medium. Then 40% off one small can because we bought 3 total, so -$4, making the total $34",  # Error in discount application
        "Let me solve this step by step:\n1. Group the purchases optimally:\n   - Group 1: Small ($10) + Medium ($18) → 30% off small = -$3\n   - Group 2: Small ($10) alone\n2. Calculate total:\n   - Group 1: $18 + $10 - $3 = $25\n   - Group 2: $10\n   - Total: $25 + $10 = $35\nTherefore, the best price is $35"
    ),
    (
        "A water tank loses 15% of its water to evaporation each day. If the tank starts with 1000 liters and we want to maintain at least 500 liters, after how many days should we refill it?",
        "Let's solve this:\n1. Each day we keep 85% of the water\n2. Day 1: 1000 × 0.85 = 850 liters\n3. Day 2: 850 × 0.85 = 722.5 liters\n4. Day 3: 722.5 - 15% = 614.125 liters, so",  # Error in calculation method
        "Let me solve this step by step:\n1. Each day remaining water = previous day × 0.85\n2. Day 1: 1000 × 0.85 = 850 liters\n3. Day 2: 850 × 0.85 = 722.5 liters\n4. Day 3: 722.5 × 0.85 = 614.13 liters\n5. Day 4: 614.13 × 0.85 = 522.01 liters\n6. Day 5: 522.01 × 0.85 = 443.71 liters (below 500)\nTherefore, the tank should be refilled after 4 days"
    ),
    (
        "In a bag of 60 marbles, the ratio of red to blue to green marbles is 2:3:1. Five marbles are drawn randomly. What's the probability that at least 3 of them are blue?",
        "Let's break this down:\n1. First, find the number of each color: 20 red, 30 blue, 10 green\n2. To get at least 3 blue, we need combinations of 3, 4, or 5 blue\n3. Total possible combinations is C(60,5)\n4. For exactly 3 blue, we need C(30,3) × C(30,2) because",  # Error in combination calculation
        "Let me solve this step by step:\n1. Total marbles = 60, with 30 blue, 20 red, 10 green\n2. Need probability of (3 blue + 2 non-blue) OR (4 blue + 1 non-blue) OR (5 blue)\n3. Calculate each case:\n   - 3 blue: C(30,3) × C(30,2) / C(60,5)\n   - 4 blue: C(30,4) × C(30,1) / C(60,5)\n   - 5 blue: C(30,5) / C(60,5)\n4. P(3 blue) = (4060 × 435) / 5461512 = 0.3230\n5. P(4 blue) = (27405 × 30) / 5461512 = 0.1506\n6. P(5 blue) = 142506 / 5461512 = 0.0261\n7. Total probability = 0.3230 + 0.1506 + 0.0261 = 0.4997\nTherefore, the probability is approximately 0.50 or 50%"
    ),
    (
        "A store sells notebooks for $4 each. Every third notebook is 25% off, and every fourth notebook is 40% off. If a customer buys 12 notebooks, how much do they spend?",
        "Let's calculate:\n1. First, identify which notebooks get discounts:\n   - Every 3rd: 3,6,9,12\n   - Every 4th: 4,8,12\n2. Notebook 12 gets both discounts, so it's 65% off\n3. Now let's count regular price ones:",  # Error in discount stacking
        "Let me solve this step by step:\n1. Identify notebooks with discounts:\n   - Every 3rd (25% off): 3,6,9,12 (4 notebooks)\n   - Every 4th (40% off): 4,8,12 (3 notebooks)\n   - Notebook 12 gets better discount (40%)\n2. Calculate prices:\n   - Regular price ($4): 1,2,5,7,10,11 (6 notebooks) = $24\n   - 25% off ($3): 3,6,9 (3 notebooks) = $9\n   - 40% off ($2.40): 4,8,12 (3 notebooks) = $7.20\nTherefore, total spent = $24 + $9 + $7.20 = $40.20"
    ),
    (
        "In a card game, you draw 3 cards from a standard 52-card deck. You win if you get exactly two hearts. If you draw the ace of hearts, you must draw again. What's the probability of winning?",
        "First, let's count favorable outcomes: We need 2 hearts out of 3 cards. There are 13 hearts, and 39 non-hearts. So it's (C(13,2) × 39) / (C(52,3)), then subtract ace of hearts cases",  # Error in probability calculation
        "Let me solve this step by step:\n1. First calculate probability without ace rule:\n   P1 = [C(12,2) × C(39,1) + C(12,1) × C(39,2)] / C(52,3)\n2. If ace of hearts is drawn:\n   - First card: P(ace♥) = 1/52\n   - Second card: P(one♥) = 12/51\n   - Third card: P(non♥) = 39/50\n   Plus permutations\n3. Total probability = P1 + P(ace cases that lead to two hearts)\n4. Calculating: 0.2254 + 0.0113 = 0.2367\nTherefore, the probability is approximately 0.2367 or 23.67%"
    ),
    (
        "A digital clock shows hours and minutes (like 23:59). How many times between 00:00 and 23:59 do exactly three identical digits appear?",
        "Let's solve systematically:\n1. We need three digits to be the same\n2. Each time has four digits (HH:MM)\n3. For hours, we can use 00-23\n4. For minutes, we can use 00-59\n5. Let's count when three 0s appear: 00:00, 00:01, 00:02, so",  # Error in counting strategy
        "Let me solve this step by step:\n1. For each digit (0-9):\n   - Check if it can appear in hours (00-23)\n   - Check if it can appear in minutes (00-59)\n   - Count combinations where exactly three digits match\n2. For digit 0:\n   00:00, 00:01, ..., 00:09, 10:00, 20:00\n3. For digit 1:\n   11:11, 11:12, ..., 11:19\n4. Continue for each digit...\n5. Total count = 180\nTherefore, there are 180 times when exactly three identical digits appear"
    ),
    (
        "A square piece of paper is folded in half vertically, then in half horizontally, creating four equal rectangles. If the perimeter of one small rectangle is 30 inches, what was the perimeter of the original square?",
        "Let's solve this:\n1. After folding, we have four equal rectangles\n2. If one rectangle's perimeter is 30 inches, let's find its dimensions\n3. The rectangle's length must be twice its width\n4. So if perimeter is 30, then length + width + length + width = 30",  # Error in dimension calculation
        "Let me solve this step by step:\n1. Each small rectangle is 1/4 of the original square\n2. If small rectangle perimeter = 30 inches:\n   2l + 2w = 30\n   l + w = 15\n3. Since it came from folding a square in half twice:\n   Original square side = 2w = 2l\n   Therefore w = l = 7.5 inches\n4. Original square perimeter = 4 × 15 = 60\nTherefore, the original square's perimeter was 60 inches"
    ),
    (
        "Three gears form a chain. The first has 12 teeth, the second has 36 teeth, and the third has 24 teeth. If the first gear makes 60 complete rotations, how many rotations does the third gear make?",
        "Let's solve this:\n1. When gears mesh, their rotations are inversely proportional to their teeth\n2. From first to second gear: 12/36 = 1/3, so second gear makes 20 rotations\n3. From second to third: 36/24 = 3/2, so third gear makes 30 rotations\n4. Therefore, when first gear makes 60 rotations, third gear makes",  # Error in gear ratio calculation
        "Let me solve this step by step:\n1. First to second gear ratio: 12:36 = 1:3\n   - When first gear makes 60 rotations\n   - Second gear makes 60 × (12/36) = 20 rotations\n2. Second to third gear ratio: 36:24 = 3:2\n   - When second gear makes 20 rotations\n   - Third gear makes 20 × (36/24) = 30 rotations\nTherefore, the third gear makes 30 complete rotations"
    ),
    (
        "A recipe calls for 3/4 cup of sugar to make 12 cookies. How much sugar is needed to make 30 cookies?",
        "To make 30 cookies, that's 2.5 times more cookies, so I'll add 2.5 to 3/4 cup to find the new amount of",  # Error: adding instead of multiplying
        "Let me solve this step by step:\n1. Current recipe: 12 cookies needs 3/4 cup sugar\n2. New amount: 30 cookies = 12 × 2.5\n3. Sugar needed = 3/4 × 2.5 = 15/8 = 1.875\nTherefore, 1.875 cups of sugar are needed"
    ),
    (
        "In a game, you score 3 points for each red target and lose 2 points for each blue target. If you hit 8 red targets and 5 blue targets, what's your final score?",
        "Let me calculate: 8 red targets is 24 points, 5 blue targets is 10 points, now I'll add these together since all points should be",  # Error: adding penalty instead of subtracting
        "Let me solve this step by step:\n1. Points from red targets: 8 × 3 = 24 points\n2. Points from blue targets: 5 × (-2) = -10 points\n3. Final score = 24 - 10 = 14\nTherefore, the final score is 14 points"
    ),
    (
        "A triangle has angles measuring x, 2x, and 3x. What is the value of x?",
        "Since it's a triangle, like a circle, x + 2x + 3x must equal 360, and now I can solve for",  # Error: using 360 instead of 180
        "Let me solve this step by step:\n1. In a triangle, angles sum to 180\n2. So: x + 2x + 3x = 180\n3. Combine like terms: 6x = 180\n4. Solve for x: x = 30\nTherefore, x = 30"
    ),
    (
        "A store sells candies in bags of 6 for $2.40. How much would 15 candies cost?",
        "If 6 candies cost $2.40, then each candy costs $2.40 ÷ 6 = $0.40. Now I'll add $0.40 repeatedly until I reach",  # Error: adding instead of multiplying unit price
        "Let me solve this step by step:\n1. Cost per candy = $2.40 ÷ 6 = $0.40\n2. Cost for 15 candies = 15 × $0.40 = $6.00\nTherefore, 15 candies would cost $6.00"
    ),
    (
        "A square room has an area of 64 square feet. If you want to place a border of tiles around the edge of the room, how many 1-foot tiles do you need?",
        "The area is 64 sq ft, so each side is 8 feet. To find the number of tiles needed around the edge, I'll multiply 8 × 8 since that's how many tiles would cover",  # Error: calculating area instead of perimeter
        "Let me solve this step by step:\n1. Area = 64 sq ft, so side length = 8 feet (since 8 × 8 = 64)\n2. Perimeter = 4 × side length = 4 × 8 = 32\n3. Each tile is 1 foot, so need one tile per foot\nTherefore, you need 32 tiles"
    ),
    (
        "At a fair, balloons come in packs of 5 for $3. If you need 23 balloons, how much will you spend?",
        "First, I'll divide 23 by 5:\n23 ÷ 5 = 4 remainder 3\nThen multiply $3 by 4 since that's how many complete packs we need. The remaining 3 balloons will be free because",  # Error: assuming remaining balloons are free
        "Let me solve this step by step:\n1. Need 23 balloons, packs contain 5 balloons\n2. Number of packs needed = ceil(23 ÷ 5) = 5 packs\n3. Cost = 5 packs × $3 = $15\nTherefore, you will spend $15"
    ),
    (
        "In a sequence, each number is found by doubling the previous number and subtracting 1. If the sequence starts with 4, what's the 4th number?",
        "Step 1: Start with 4\nStep 2: Double it to get 8, subtract 1 to get 7\nStep 3: Double 7 to get 14, subtract 1 to get 13\nStep 4: Now I'll double 13 and",  # Error: correct process but unfinished
        "Let me solve this step by step:\n1. Start: 4\n2. Next: (4 × 2) - 1 = 7\n3. Then: (7 × 2) - 1 = 13\n4. Finally: (13 × 2) - 1 = 25\nTherefore, the 4th number is 25"
    ),
    (
        "A cylindrical water tank has a radius of 3 feet and height of 4 feet. How many cubic feet of water can it hold?",
        "> Starting with radius = 3 feet\n> Area of base = π × radius squared\n> So area = 3.14 × 3 × 3\n> Now multiply by height because",  # Error: using 3.14 instead of π and showing calculation
        "Let me solve this step by step:\n1. Volume = π × radius² × height\n2. Radius = 3 feet\n3. Height = 4 feet\n4. Volume = π × 3² × 4 = 113.1 cubic feet\nTherefore, the tank can hold 113.1 cubic feet of water"
    ),
    (
        "If 15% of a number is 45, what is 25% of the same number?",
        "1) First find the number:\n   - 15% means multiply by 0.15\n   - So if 0.15x = 45\n2) Now add 10% to 45 since 25% is 10% more than 15%, which means",  # Error: adding percentage difference instead of solving proportionally
        "Let me solve this step by step:\n1. Let x be the number\n2. 15% of x = 45\n3. 0.15x = 45\n4. x = 300\n5. 25% of x = 0.25 × 300 = 75\nTherefore, 25% of the number is 75"
    ),
    (
        "Three friends split a bill of $85. The first pays 20% more than the second, who pays 20% more than the third. How much does each person pay?",
        "Let's work backwards:\n1) Total is $85\n2) First pays 20% more than second\n3) Second pays 20% more than third\n4) So if we divide $85 by 3 to get the average, then adjust by 20% for each person, that means",  # Error: dividing equally first
        "Let me solve this step by step:\n1. Let third person's payment be x\n2. Second pays 1.2x\n3. First pays 1.2(1.2x) = 1.44x\n4. Total equation: x + 1.2x + 1.44x = 85\n5. 3.64x = 85\n6. x = 23.35\n7. Second pays: 28.02\n8. First pays: 33.63\nTherefore, they pay $33.63, $28.02, and $23.35"
    )
]