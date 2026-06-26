import pytest
from typing import Callable
from hearthstone.engine.enums import SpellIDs, CardIDs, Tags, UnitType
from hearthstone.engine.entities import Spell, HandCard, Player, StoreItem, Unit
from hearthstone.engine.game import Game
from hearthstone.engine.pool import SpellPool
from hearthstone.engine.tavern import TavernManager

class TestAllSpells:
    def test_spell_pool_initialization(self) -> None:
        pool = SpellPool()
        # Verify that all our enabled spells are in the pool
        assert SpellIDs.TAVERN_COIN in pool.tiers[1]
        assert SpellIDs.BANANA in pool.tiers[1]
        assert SpellIDs.POINTY_ARROW in pool.tiers[1]
        assert SpellIDs.FORTIFY in pool.tiers[1]
        assert SpellIDs.APPLE in pool.tiers[1]
        
        assert SpellIDs.LEAF_THROUGH_THE_PAGES in pool.tiers[2]
        
        assert SpellIDs.BLOOD_GEM_BARRAGE in pool.tiers[3]
        assert SpellIDs.HAUNTED_CARAPACE in pool.tiers[3]
        assert SpellIDs.SHINY_RING in pool.tiers[3]
        assert SpellIDs.MOUNTING_AVALANCHE in pool.tiers[3]
        
        assert SpellIDs.GEM_CONFISCATION in pool.tiers[4]
        assert SpellIDs.STAFF_OF_ENRICHMENT in pool.tiers[4]
        assert SpellIDs.MISPLACED_TEA_SET in pool.tiers[4]
        assert SpellIDs.TOMB_TURNING in pool.tiers[4]
        
        assert SpellIDs.BARGAIN_BUNDLE in pool.tiers[5]
        assert SpellIDs.WAVE_OF_GOLD in pool.tiers[5]
        
        assert SpellIDs.AZERITE_EMPOWERMENT in pool.tiers[6]
        assert SpellIDs.PERFECT_VISION in pool.tiers[6]

    def test_draw_spells_respects_tier(self) -> None:
        pool = SpellPool()
        # Draw T1 spells
        drawn_t1 = pool.draw_spells(50, max_tier=1)
        for sid in drawn_t1:
            assert sid in [SpellIDs.TAVERN_COIN, SpellIDs.BANANA, SpellIDs.POINTY_ARROW, SpellIDs.FORTIFY, SpellIDs.APPLE]
            
        # Draw T2 spells (can include T1)
        drawn_t2 = pool.draw_spells(50, max_tier=2)
        assert SpellIDs.LEAF_THROUGH_THE_PAGES in drawn_t2

    def test_buy_and_cast_leaf_through_the_pages(self, empty_game: Game, player: Player, tavern: TavernManager) -> None:
        player.gold = 10
        player.free_refreshes = 0
        player.store.clear()
        spell = Spell.create_from_db(SpellIDs.LEAF_THROUGH_THE_PAGES)
        player.store.append(StoreItem(spell=spell))
        
        # Buy
        success, info = tavern.buy_unit(player, 0)
        assert success
        assert player.gold == 9  # cost is 1
        assert len(player.hand) == 1
        assert player.hand[0].spell == spell
        
        # Cast
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        assert player.free_refreshes == 2
        assert len(player.hand) == 0
        
        # Roll uses a free refresh
        success, info = tavern.roll_tavern(player)
        assert success
        assert player.free_refreshes == 1
        assert player.gold == 9  # no gold spent!

    def test_cast_blood_gem_barrage(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        # Add some board units
        u1 = mock_unit(CardIDs.MICROBOT)  # 1/1
        u2 = mock_unit(CardIDs.ROT_HIDE_GNOLL)  # 1/4
        player.board = [u1, u2]
        
        spell = Spell.create_from_db(SpellIDs.BLOOD_GEM_BARRAGE)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        
        assert u1.cur_atk == 2 and u1.cur_hp == 2
        assert u2.cur_atk == 2 and u2.cur_hp == 5

    def test_cast_gem_confiscation(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)  # 1/1
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.GEM_CONFISCATION)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        # Cast on target index 0
        success, info = tavern.play_unit(player, hand_index=0, target_index=0)
        assert success
        
        assert u.cur_atk == 2 and u.cur_hp == 2

    def test_cast_haunted_carapace(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)  # 1/1
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.HAUNTED_CARAPACE)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        
        assert u.cur_atk == 4 and u.cur_hp == 2

    def test_cast_staff_of_enrichment(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        # Populate store
        u_store = mock_unit(CardIDs.MICROBOT)
        player.store.clear()
        player.store.append(StoreItem(unit=u_store))
        
        spell = Spell.create_from_db(SpellIDs.STAFF_OF_ENRICHMENT)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        
        assert u_store.cur_atk == 3 and u_store.cur_hp == 3

    def test_cast_shiny_ring(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.SHINY_RING)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        
        assert u.cur_atk == 2 and u.cur_hp == 2

    def test_cast_mounting_avalanche(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.MOUNTING_AVALANCHE)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=0)
        assert success
        
        assert u.cur_atk == 3 and u.cur_hp == 3

    def test_cast_misplaced_tea_set(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        # Add minions of different types
        u_beast = mock_unit(CardIDs.MANASABER)  # beast
        u_mech = mock_unit(CardIDs.MICROBOT)  # mech
        player.board = [u_beast, u_mech]
        
        spell = Spell.create_from_db(SpellIDs.MISPLACED_TEA_SET)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        
        assert u_beast.cur_atk == u_beast.base_atk + 2
        assert u_mech.cur_atk == u_mech.base_atk + 2

    def test_cast_tomb_turning(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.TOMB_TURNING)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=0)
        assert success
        
        assert u.cur_atk == 3 and u.cur_hp == 3

    def test_cast_bargain_bundle(self, empty_game: Game, player: Player, tavern: TavernManager) -> None:
        player.gold = 10
        spell = Spell.create_from_db(SpellIDs.BARGAIN_BUNDLE)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        assert not player.is_discovering
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        assert player.is_discovering
        assert player.discovery.discover_tier == 5
        assert player.discovery.is_exact_tier

    def test_cast_wave_of_gold(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.WAVE_OF_GOLD)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        
        assert u.cur_atk == 4 and u.cur_hp == 3

    def test_cast_azerite_empowerment(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.AZERITE_EMPOWERMENT)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=-1)
        assert success
        
        assert u.cur_atk == 5 and u.cur_hp == 5

    def test_cast_perfect_vision(self, empty_game: Game, player: Player, tavern: TavernManager, mock_unit: Callable[..., Unit]) -> None:
        u = mock_unit(CardIDs.MICROBOT)  # base is 1/1
        player.board = [u]
        
        spell = Spell.create_from_db(SpellIDs.PERFECT_VISION)
        player.hand.append(HandCard(uid=1111, spell=spell))
        
        success, info = tavern.play_unit(player, hand_index=0, target_index=0)
        assert success
        
        assert u.cur_atk == 20
        assert u.cur_hp == 20
