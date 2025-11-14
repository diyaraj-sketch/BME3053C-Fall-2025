local Paddle = require "paddle"
local Ball = require "ball"

local left, right, ball
local font, smallFont
local paused = false
local winScore = 10
local winner = nil
local singlePlayer = true
local aiSkill = 0.92 -- 0..1, higher is harder
local menuOpen = false
local dragging = false

local function clamp(v, a, b)
  if v < a then return a end
  if v > b then return b end
  return v
end

function love.load()
  math.randomseed(os.time())
  font = love.graphics.newFont(32)
  smallFont = love.graphics.newFont(14)
  love.window.setTitle("Pong")
  local w,h = love.graphics.getWidth(), love.graphics.getHeight()
  left = Paddle.new(30, h/2 - 40, 10, 80, 400)
  right = Paddle.new(w - 40, h/2 - 40, 10, 80, 400)
  ball = Ball.new(w/2, h/2, 7, 250)
end

function love.update(dt)
  if winner then return end
  if paused then return end
  left:update(dt, "w", "s")
  if singlePlayer then
    right:aiUpdate(dt, ball, aiSkill)
  else
    right:update(dt, "up", "down")
  end
  -- allow keyboard adjustments while menu is open
  if menuOpen then
    if love.keyboard.isDown("left") then
      aiSkill = clamp(aiSkill - 0.02, 0, 1)
    end
    if love.keyboard.isDown("right") then
      aiSkill = clamp(aiSkill + 0.02, 0, 1)
    end
  end
  local scored = ball:update(dt, left, right)
  if scored == 'left' then left.score = left.score + 1; ball:reset(love.graphics.getWidth()/2, love.graphics.getHeight()/2) end
  if scored == 'right' then right.score = right.score + 1; ball:reset(love.graphics.getWidth()/2, love.graphics.getHeight()/2) end
  if left.score >= winScore then winner = "Left Player" end
  if right.score >= winScore then winner = "Right Player" end
end

function love.draw()
  love.graphics.setFont(font)
  love.graphics.printf(left.score, 0, 20, love.graphics.getWidth()/2, "center")
  love.graphics.printf(right.score, love.graphics.getWidth()/2, 20, love.graphics.getWidth()/2, "center")
  love.graphics.setFont(smallFont)
  local modeText = singlePlayer and "Single Player (AI) — press Tab to toggle" or "Two Player — press Tab to toggle"
  love.graphics.printf("W/S  vs  Up/Down — Press Space to pause, R to reset", 0, love.graphics.getHeight() - 42, love.graphics.getWidth(), "center")
  love.graphics.printf(modeText, 0, love.graphics.getHeight() - 24, love.graphics.getWidth(), "center")
  left:draw()
  right:draw()
  ball:draw()
  if paused then
    love.graphics.printf("PAUSED", 0, love.graphics.getHeight()/2 - 20, love.graphics.getWidth(), "center")
  end
  if winner then
    love.graphics.printf(winner .. " Wins! Press R to play again.", 0, love.graphics.getHeight()/2 - 60, love.graphics.getWidth(), "center")
  end

  -- Difficulty menu / slider
  love.graphics.setFont(smallFont)
  love.graphics.printf("Press M to open Difficulty Menu (drag or use ← →)", 8, love.graphics.getHeight() - 60, love.graphics.getWidth(), "left")
  if menuOpen then
    local sw, sh = 380, 90
    local sx, sy = (love.graphics.getWidth() - sw) / 2, (love.graphics.getHeight() - sh) / 2
    -- panel
    love.graphics.setColor(0, 0, 0, 0.7)
    love.graphics.rectangle("fill", sx, sy, sw, sh, 6, 6)
    love.graphics.setColor(1,1,1,1)
    love.graphics.printf("Difficulty", sx, sy + 8, sw, "center")

    -- slider
    local sliderX, sliderY, sliderW, sliderH = sx + 24, sy + 42, sw - 48, 10
    love.graphics.setColor(0.6, 0.6, 0.6, 1)
    love.graphics.rectangle("fill", sliderX, sliderY, sliderW, sliderH, 4, 4)
    love.graphics.setColor(0.2, 0.8, 0.2, 1)
    local fillW = aiSkill * sliderW
    love.graphics.rectangle("fill", sliderX, sliderY, fillW, sliderH, 4, 4)
    -- knob
    local knobX = sliderX + fillW
    love.graphics.setColor(1,1,1,1)
    love.graphics.circle("fill", knobX, sliderY + sliderH/2, 8)

    love.graphics.printf(string.format("AI Skill: %.2f", aiSkill), sx, sy + 60, sw, "center")
  end
  love.graphics.setColor(1,1,1,1)
end

function love.keypressed(key)
  if key == "space" then paused = not paused end
  if key == "r" then
    left.score = 0; right.score = 0; winner = nil; ball:reset(love.graphics.getWidth()/2, love.graphics.getHeight()/2)
  end
  if key == "tab" then
    singlePlayer = not singlePlayer
  end
  if key == "m" then
    menuOpen = not menuOpen
  end
  if key == "escape" then love.event.quit() end
end

function love.mousepressed(x, y, b)
  if menuOpen and b == 1 then
    local sw, sh = 380, 90
    local sx, sy = (love.graphics.getWidth() - sw) / 2, (love.graphics.getHeight() - sh) / 2
    local sliderX, sliderY, sliderW, sliderH = sx + 24, sy + 42, sw - 48, 10
    if x >= sliderX and x <= sliderX + sliderW and y >= sliderY - 10 and y <= sliderY + sliderH + 10 then
      local rel = (x - sliderX) / sliderW
      aiSkill = clamp(rel, 0, 1)
      dragging = true
    end
  end
end

function love.mousereleased(x, y, b)
  if b == 1 then dragging = false end
end

function love.mousemoved(x, y, dx, dy)
  if menuOpen and dragging then
    local sw, sh = 380, 90
    local sx, sy = (love.graphics.getWidth() - sw) / 2, (love.graphics.getHeight() - sh) / 2
    local sliderX, sliderY, sliderW, sliderH = sx + 24, sy + 42, sw - 48, 10
    local rel = (x - sliderX) / sliderW
    aiSkill = clamp(rel, 0, 1)
  end
end
