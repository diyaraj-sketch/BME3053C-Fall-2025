local Paddle = {}
Paddle.__index = Paddle

function Paddle.new(x, y, w, h, speed)
  local self = setmetatable({}, Paddle)
  self.x = x
  self.y = y
  self.w = w
  self.h = h
  self.speed = speed or 300
  self.score = 0
  return self
end

function Paddle:update(dt, upKey, downKey)
  if love.keyboard.isDown(upKey) then
    self.y = self.y - self.speed * dt
  end
  if love.keyboard.isDown(downKey) then
    self.y = self.y + self.speed * dt
  end
  local screenH = love.graphics.getHeight()
  if self.y < 0 then self.y = 0 end
  if self.y + self.h > screenH then self.y = screenH - self.h end
end

function Paddle:draw()
  love.graphics.rectangle("fill", self.x, self.y, self.w, self.h)
end

function Paddle:centerY()
  return self.y + self.h / 2
end

-- AI: move paddle toward the ball with an adjustable skill (0..1)
function Paddle:aiUpdate(dt, ball, skill)
  skill = skill or 0.9
  -- Predict where the ball will be vertically when it reaches this paddle's x
  local predictY = ball.y
  if ball.vx ~= 0 then
    local timeToReach = math.abs((self.x - ball.x) / ball.vx)
    predictY = ball.y + ball.vy * timeToReach
  end

  -- clamp prediction inside screen
  local sh = love.graphics.getHeight()
  if predictY < 0 then predictY = 0 end
  if predictY > sh then predictY = sh end

  local tolerance = 6
  local target = predictY
  if self:centerY() < target - tolerance then
    self.y = self.y + self.speed * skill * dt
  elseif self:centerY() > target + tolerance then
    self.y = self.y - self.speed * skill * dt
  end

  -- clamp
  if self.y < 0 then self.y = 0 end
  if self.y + self.h > sh then self.y = sh - self.h end
end

return Paddle
