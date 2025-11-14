local Ball = {}
Ball.__index = Ball

function Ball.new(x, y, r, speed)
  local self = setmetatable({}, Ball)
  self.x = x
  self.y = y
  self.r = r or 6
  self.speed = speed or 250
  local angle = math.rad(math.random(30, 150))
  local dir = (math.random(0, 1) == 0) and -1 or 1
  self.vx = self.speed * math.cos(angle) * dir
  self.vy = self.speed * math.sin(angle) * ((math.random(0,1)==0) and -1 or 1)
  return self
end

function Ball:update(dt, leftP, rightP)
  self.x = self.x + self.vx * dt
  self.y = self.y + self.vy * dt
  local sh = love.graphics.getHeight()
  if self.y - self.r < 0 then self.y = self.r; self.vy = -self.vy end
  if self.y + self.r > sh then self.y = sh - self.r; self.vy = -self.vy end

  local function intersects(p)
    return self.x - self.r < p.x + p.w and self.x + self.r > p.x and self.y + self.r > p.y and self.y - self.r < p.y + p.h
  end

  if intersects(leftP) and self.vx < 0 then
    self.x = leftP.x + leftP.w + self.r
    self.vx = -self.vx * 1.03
    local offset = (self.y - leftP:centerY()) / (leftP.h / 2)
    self.vy = self.vy + offset * 150
  end

  if intersects(rightP) and self.vx > 0 then
    self.x = rightP.x - self.r
    self.vx = -self.vx * 1.03
    local offset = (self.y - rightP:centerY()) / (rightP.h / 2)
    self.vy = self.vy + offset * 150
  end

  if self.x < 0 then return 'right' end
  if self.x > love.graphics.getWidth() then return 'left' end
  return nil
end

function Ball:reset(x, y)
  self.x = x
  self.y = y
  self.speed = 250
  local angle = math.rad(math.random(30, 150))
  local dir = (math.random(0, 1) == 0) and -1 or 1
  self.vx = self.speed * math.cos(angle) * dir
  self.vy = self.speed * math.sin(angle) * ((math.random(0,1)==0) and -1 or 1)
end

function Ball:draw()
  love.graphics.circle("fill", self.x, self.y, self.r)
end

return Ball
